"""
Film Festival Network Analysis: Module

we write our code here as we don't want to clutter the jupyter notebook

this module tries to capture the "network effects" of the film festival circuit,
or "festivalization" as social scientific and media scholars attend to it as.
Engineers featuers and visualizations using:
1. Bipartite PageRank (FilmRank) - Recursive prestige measurement
"""
# graphing imports

from tqdm import tqdm


import numpy as np
import pandas as pd
import networkx as nx
from networkx.algorithms import bipartite
from networkx.algorithms import community as nx_community
from collections import defaultdict, Counter
import warnings
from pathlib import Path
import random
import colorsys

# visualization imports
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgba
from matplotlib.cm import ScalarMappable
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


# DATA LOADING SECTION

def loadCSV(filepath):
    """
    load a CSV file with automatic delimiter detection.
    
    handles the Film Circulation dataset files:
    - Most files use comma (,) delimiter
    - '4_festival-library_dataset_imdb-and-survey.csv' uses semicolon (;) delimiter
    """
    
    
    
    # Check if this is the festival library file (uses semicolon)
    if 'festival-library' in filepath or 'festival_library' in filepath:
        df = pd.read_csv(filepath, sep=';', low_memory=False)
        
        return df
    
    # For other files, try comma first, then semicolon
    try:
        df = pd.read_csv(filepath, sep=',', low_memory=False)
        if len(df.columns) > 1:
            
            return df
    except:
        pass
    
    # Try semicolon
    try:
        df = pd.read_csv(filepath, sep=';', low_memory=False)
        
        return df
    except Exception as e:
        raise ValueError(f"Could not load {filepath}: {e}")


def loadFilmCirculationData(dataDir='./data'):
    """
    load all Film Circulation dataset files from a directory.
    """
    dataPath = Path(dataDir)
    
    festivalRuns = None
    filmMetadata = None
    festivalMetadata = None
    
    for fname in ['3_imdb_dataset_festival-runs_long.csv', '1_film-dataset_festival-program_long.csv']:
        fpath = dataPath / fname
        if fpath.exists():
            festivalRuns = loadCSV(str(fpath))
            break
    
    for fname in ['merged_film_data.csv']:
        fpath = dataPath / fname
        if fpath.exists():
            filmMetadata = loadCSV(str(fpath))
            break
    
    for fname in ['4_festival-library_dataset_imdb-and-survey.csv']:
        fpath = dataPath / fname
        if fpath.exists():
            festivalMetadata = loadCSV(str(fpath))
            break
    
    
    
    
    
    return festivalRuns, filmMetadata, festivalMetadata


# MAIN GRAPH BUILDING SECTIONS

def createBipartiteGraph( festivalRuns, filmIdCol='unique.id', 
                         festivalIdCol='festival.id', yearCol='year',
                         cutoffYear=None, weightByAppearances=True):
    """
    construct a bipartite graph from festival run data. (for filmrank)
    """
    
    data = festivalRuns.copy()
    if cutoffYear is not None:
        data = data[data[yearCol] <= cutoffYear]
    
    G = nx.Graph()
    
    if weightByAppearances:
        edgeCounts = data.groupby([filmIdCol, festivalIdCol]).size().reset_index(name='weight')
    else:
        edgeCounts = data[[filmIdCol, festivalIdCol]].drop_duplicates()
        edgeCounts['weight'] = 1
    
    films = set()
    festivals = set()
    
    for _, row in edgeCounts.iterrows():
        filmId = row[filmIdCol]
        festivalId = row[festivalIdCol]
        weight = row['weight']
        
        filmNode = f"film_{filmId}"
        festNode = f"fest_{festivalId}"
        
        if filmNode not in films:
            G.add_node(filmNode, bipartite=0, node_type='film', original_id=filmId)
            films.add(filmNode)
        
        if festNode not in festivals:
            G.add_node(festNode, bipartite=1, node_type='festival', original_id=festivalId)
            festivals.add(festNode)
        
        G.add_edge(filmNode, festNode, weight=weight)
    
    
    
    
    
    return G


def createfilmOnly(bipartiteGraph):
    """
    create a film-only graph where films are connected if they share festivals.
    edge weight = number of shared festivals.
    """
    filmNodes = [n for n in bipartiteGraph.nodes() if bipartiteGraph.nodes[n].get('node_type') == 'film']
    
    # create projection (film-only is projection on previous graph)
    filmGraph = nx.Graph()
    
    for node in filmNodes:
        filmGraph.add_node(node, **bipartiteGraph.nodes[node])
    
    # connect films that share festivals
    for i, film1 in enumerate(filmNodes):
        festivals1 = set(bipartiteGraph.neighbors(film1))
        for film2 in filmNodes[i+1:]:
            festivals2 = set(bipartiteGraph.neighbors(film2))
            shared = festivals1 & festivals2
            if shared:
                filmGraph.add_edge(film1, film2, weight=len(shared), shared_festivals=list(shared))
    
    
    
    return filmGraph


# CALCULATING PAGE RANK
def computeBipartiteFilmRank(bipartiteGraph,alpha=0.85,maxIter=100,tol=1e-6
):
    """
    Compute FilmRank (pagerank) scores for the bipartite film-festival graph.
    """
    
    filmranks = nx.pagerank(bipartiteGraph, alpha=alpha, max_iter=maxIter, tol=tol, weight='weight')
    
    filmFilmRanks = {}
    festivalFilmRanks = {}
    
    for node, score in filmranks.items():
        originalId = bipartiteGraph.nodes[node].get('original_id', node)
        if bipartiteGraph.nodes[node].get('bipartite') == 0:
            filmFilmRanks[originalId] = score
        else:
            festivalFilmRanks[originalId] = score
    
    # normalize to [0, 1]
    if filmFilmRanks:
        maxFilm = max(filmFilmRanks.values())
        if maxFilm > 0:
            filmFilmRanks = {k: v / maxFilm for k, v in filmFilmRanks.items()}
    
    if festivalFilmRanks:
        maxFest = max(festivalFilmRanks.values())
        if maxFest > 0:
            festivalFilmRanks = {k: v / maxFest for k, v in festivalFilmRanks.items()}

    
    return filmFilmRanks, festivalFilmRanks


def computeTemporalFilmRank(festivalRuns,filmIdCol='unique.id',festivalIdCol='festival.id',yearCol='year',
    yearsToCompute=None
):
    """
    compute FilmRank at multiple time points for temporal features.
    """
    if yearsToCompute is None:
        yearsToCompute = sorted(festivalRuns[yearCol].unique())
    
    allResults = []
    
    for year in yearsToCompute:
        
        
        G = createBipartiteGraph(
            festivalRuns, filmIdCol, festivalIdCol, yearCol,
            cutoffYear=year, weightByAppearances=True
        )
        
        if G.number_of_edges() < 5:
            continue
        
        filmPR, festPR = computeBipartiteFilmRank(G)
        
        for filmId, score in filmPR.items():
            allResults.append({
                'film_id': filmId,
                'year': year,
                'filmrank': score
            })
    
    resultsDf = pd.DataFrame(allResults)
    
    return resultsDf


# VISUALIZTATION HELPERS

def detectCommunities(graph, resolution=1.0, seed=42):
    """detect communities using Louvain algorithm.
    """
    random.seed(seed)
    np.random.seed(seed)
    
    try:
        communities = nx_community.louvain_communities(graph, resolution=resolution, seed=seed)
    except:
        communities = list(nx_community.greedy_modularity_communities(graph))
    
    nodeToCommunity = {}
    for commId, commNodes in enumerate(communities):
        for node in commNodes:
            nodeToCommunity[node] = commId
    
    return nodeToCommunity


def generateCommunityColors(n):
    """generate n community distinct colors for viz
    """
    colors = []
    for i in range(n):
        hue = i / n
        r, g, b = colorsys.hls_to_rgb(hue, 0.5, 0.7)
        colors.append(f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}')
    return colors


def labelCommunity(nodes, graph, filmMetadata, festivalMetadata, filmIdCol):
    """
    generate a descriptive label for a community based on member characteristics.
    """
    metaLookup = {}
    if filmMetadata is not None:
        for _, row in filmMetadata.iterrows():
            fid = row.get(filmIdCol)
            if fid is not None:
                metaLookup[fid] = row
    
    festLookup = {}
    if festivalMetadata is not None:
        for _, row in festivalMetadata.iterrows():
            fid = row.get('festival.id')
            if fid is not None:
                festLookup[fid] = row
    
    counts = {'doc': 0, 'fict': 0, 'animt': 0, 'exp': 0, 'western': 0, 'non_western': 0, 'lgbtq': 0, 
              'film_total': 0, 'fest_total': 0}
    
    for node in nodes:
        nodeType = graph.nodes[node].get('node_type', 'unknown')
        originalId = graph.nodes[node].get('original_id')
        
        if nodeType == 'film':
            counts['film_total'] += 1
            if originalId in metaLookup:
                meta = metaLookup[originalId]
                for genre in ['doc', 'fict', 'animt', 'exp']:
                    genreVal = meta.get(genre, 0)
                    if pd.notna(genreVal) and float(genreVal) == 1.0:
                        counts[genre] += 1
                
                if (meta.get('regions.eu', 0) == 1) or (meta.get('regions.na', 0) == 1):
                    counts['western'] += 1
                else:
                    counts['non_western'] += 1
                
                lgbtqVal = meta.get('lgbtq', '')
                if isinstance(lgbtqVal, str) and 'lgbtq' in lgbtqVal.lower():
                    counts['lgbtq'] += 1
        
        elif nodeType == 'festival':
            counts['fest_total'] += 1
    
    # build label
    labels = []
    
    total = counts['film_total']
    if total > 0:
        for genre, name in [('doc', 'Doc'), ('fict', 'Fiction'), ('animt', 'Animation'), ('exp', 'Experimental')]:
            if counts[genre] / total > 0.4:
                labels.append(name)
        
        if counts['non_western'] / total > 0.3:
            labels.append('Intl')
        elif counts['western'] / total > 0.8:
            labels.append('Western')
        
        if counts['lgbtq'] / total > 0.2:
            labels.append('LGBTQ+')
    
    if counts['fest_total'] > 0 and counts['film_total'] == 0:
        return f"Festivals (n={counts['fest_total']})"
    
    if not labels:
        labels.append('Mixed')
    
    return ' / '.join(labels)


def computeLayoutWithCommunities(graph, nodeToCommunity, seed=42):
    """
    compute a layout that groups nodes by community.
    """
    communities = sorted(set(nodeToCommunity.values()))
    nComm = len(communities)
    
    # position community centroids
    commGraph = nx.Graph()
    for c in communities:
        commGraph.add_node(c)
    
    commEdges = defaultdict(float)
    for u, v in graph.edges():
        cu, cv = nodeToCommunity.get(u), nodeToCommunity.get(v)
        if cu is not None and cv is not None and cu != cv:
            commEdges[tuple(sorted([cu, cv]))] += 1
    
    for (cu, cv), w in commEdges.items():
        commGraph.add_edge(cu, cv, weight=w)
    
    commCentroids = nx.spring_layout(commGraph, k=4, iterations=100, seed=seed)
    scale = 4 * np.sqrt(nComm)
    commCentroids = {c: (x * scale, y * scale) for c, (x, y) in commCentroids.items()}
    
    # position nodes within communities
    positions = {}
    for commId in communities:
        commNodes = [n for n, c in nodeToCommunity.items() if c == commId and n in graph]
        if not commNodes:
            continue
        
        centroid = commCentroids.get(commId, (0, 0))
        
        if len(commNodes) == 1:
            positions[commNodes[0]] = centroid
        else:
            subgraph = graph.subgraph(commNodes).copy()
            localPos = nx.spring_layout(subgraph, k=0.4, iterations=50, seed=seed + commId)
            
            coords = np.array(list(localPos.values()))
            coords -= coords.mean(axis=0)
            maxR = np.max(np.sqrt(np.sum(coords**2, axis=1))) or 1
            targetR = 0.4 * np.sqrt(len(commNodes)) * 0.8
            coords = coords * targetR / maxR
            
            for node, (lx, ly) in zip(localPos.keys(), coords):
                positions[node] = (centroid[0] + lx, centroid[1] + ly)
    
    return positions, commCentroids


def buildMetadataLookup(filmMetadata, filmIdCol):
    """
    build a lookup dict for fast metadata access
    """
    if filmMetadata is None:
        return {}
    
    lookup = {}
    for _, row in filmMetadata.iterrows():
        fid = row.get(filmIdCol)
        if fid is not None:
            lookup[fid] = row
    return lookup


#MAIN VISUALIZATION STEP

def visualizeBipartiteFilmRank( bipartiteGraph, filmFilmRanks, festivalFilmRanks,filmMetadata=None,
                               festivalMetadata=None, filmIdCol='unique.id', 
                               outputPath='viz_bipartite_pagerank.png', figsize=(24, 20)):
    """
    bipartite network visualization for FilmRank
    - Node SIZE = FilmRank score
    - Node COLOR = Community membership
    - Both films and festivals shown
    - Non-western/international films highlighted with diamond marker
    """
    
    # build metadata lookup for non-western detection
    metaLookup = buildMetadataLookup(filmMetadata, filmIdCol)
    
    # detect communities
    nodeToCommunity = detectCommunities(bipartiteGraph, resolution=1.5)
    communities = sorted(set(nodeToCommunity.values()))
    nComm = len(communities)
    colors = generateCommunityColors(nComm)
    commToColor = {c: colors[i] for i, c in enumerate(communities)}
    
    # compute layout
    positions, commCentroids = computeLayoutWithCommunities(bipartiteGraph, nodeToCommunity)
    
    # create figure
    fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    
    # draw edges
    edgeSegments = []
    edgeColors = []
    for u, v in bipartiteGraph.edges():
        if u not in positions or v not in positions:
            continue
        edgeSegments.append([positions[u], positions[v]])
        cu, cv = nodeToCommunity.get(u), nodeToCommunity.get(v)
        if cu == cv:
            edgeColors.append(to_rgba(commToColor[cu], alpha=0.15))
        else:
            edgeColors.append(to_rgba('#cccccc', alpha=0.05))
    
    if edgeSegments:
        lc = LineCollection(edgeSegments, colors=edgeColors, linewidths=0.4, zorder=1)
        ax.add_collection(lc)
    
    # draw nodes - separate western and non-western films for different markers
    for node in bipartiteGraph.nodes():
        if node not in positions:
            continue
        
        x, y = positions[node]
        nodeType = bipartiteGraph.nodes[node].get('node_type', 'unknown')
        originalId = bipartiteGraph.nodes[node].get('original_id')
        commId = nodeToCommunity.get(node, 0)
        color = commToColor[commId]
        
        if nodeType == 'film':
            pr = filmFilmRanks.get(originalId, 0)
            size = 15 + pr * 150
            
            # check if non-western - use diamond marker with dark edge
            if originalId in metaLookup:
                meta = metaLookup[originalId]
                isEU = meta.get('regions.eu', 0)
                isNA = meta.get('regions.na', 0)
                isWestern = (pd.notna(isEU) and int(isEU) == 1) or (pd.notna(isNA) and int(isNA) == 1)
                
                if not isWestern:
                    # non-western: use diamond marker with black edge for visibility
                    ax.scatter(x, y, s=size*1.3, c=color, marker='D', alpha=0.9, 
                              edgecolors='black', linewidths=0.8, zorder=3)
                else:
                    # western: circle
                    ax.scatter(x, y, s=size, c=color, marker='o', alpha=0.85, 
                              edgecolors='white', linewidths=0.3, zorder=2)
            else:
                # no metadata: default circle
                ax.scatter(x, y, s=size, c=color, marker='o', alpha=0.85, 
                          edgecolors='white', linewidths=0.3, zorder=2)
        else:
            # festival: square
            pr = festivalFilmRanks.get(originalId, 0)
            size = 30 + pr * 200
            ax.scatter(x, y, s=size, c=color, marker='s', alpha=0.85, 
                      edgecolors='white', linewidths=0.3, zorder=2)
    
    # add community labels (high zorder to appear on top of all nodes)
    for commId in communities:
        commNodes = [n for n, c in nodeToCommunity.items() if c == commId and n in positions]
        if len(commNodes) < 5:
            continue
        
        cx = np.mean([positions[n][0] for n in commNodes])
        cy = np.mean([positions[n][1] for n in commNodes])
        
        label = labelCommunity(commNodes, bipartiteGraph, filmMetadata, festivalMetadata, filmIdCol)
        
        ax.annotate(f"{label}", (cx, cy), fontsize=8, ha='center', va='center',
                   fontweight='bold', color='#2c3e50', zorder=10,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                            edgecolor=commToColor[commId], alpha=0.95, linewidth=2))
    
    # build legend with community colors and counts
    legendElements = []
    
    # sort communities by size for legend
    commSizes = {}
    for commId in communities:
        commNodes = [n for n, c in nodeToCommunity.items() if c == commId]
        commSizes[commId] = len(commNodes)
    
    sortedComms = sorted(commSizes.keys(), key=lambda x: commSizes[x], reverse=True)
    
    for commId in sortedComms:
        commNodes = [n for n, c in nodeToCommunity.items() if c == commId and n in positions]
        if len(commNodes) < 3:
            continue
        label = labelCommunity(commNodes, bipartiteGraph, filmMetadata, festivalMetadata, filmIdCol)
        legendElements.append(
            mpatches.Patch(facecolor=commToColor[commId], label=f'{label} (n={len(commNodes)})')
        )
    
    # add marker legend
    legendElements.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
                                  markersize=8, label='Film (Western)'))
    legendElements.append(Line2D([0], [0], marker='D', color='w', markerfacecolor='gray', 
                                  markeredgecolor='black', markersize=8, label='Film (Non-Western/Intl)'))
    legendElements.append(Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', 
                                  markersize=8, label='Festival'))
    
    ax.legend(handles=legendElements, loc='center right', fontsize=8, title='Communities',
             bbox_to_anchor=(1.02, 0.5))
    
    ax.set_title('Film-Festival Network: FILMRANK Community Structure\nColor = Community | Labels = Dominant Characteristics',
                fontsize=14, fontweight='bold')
    ax.axis('off')
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(outputPath, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()



def visualizeFilmsOnlyFilmRank(bipartiteGraph,filmFilmRanks,filmMetadata=None,filmIdCol='unique.id',
                               outputPath='viz_films_pagerank.png', figsize=(24, 20)
):
    """
    film-only graph/network: FilmRank
    - Films connected if they share festivals
    - Node SIZE = FilmRank score
    - Node COLOR = Community
    - Non-western/international films highlighted with diamond marker
    """
    # create film projection
    filmGraph = createfilmOnly(bipartiteGraph)
    
    # build metadata lookup for non-western detection
    metaLookup = buildMetadataLookup(filmMetadata, filmIdCol)
    
    # detect communities on film graph
    nodeToCommunity = detectCommunities(filmGraph, resolution=1.0)
    communities = sorted(set(nodeToCommunity.values()))
    nComm = len(communities)
    colors = generateCommunityColors(nComm)
    commToColor = {c: colors[i] for i, c in enumerate(communities)}
    
    # compute layout
    positions, commCentroids = computeLayoutWithCommunities(filmGraph, nodeToCommunity)
    
    # create figure
    fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    
    # draw edges (films sharing festivals)
    edgeSegments = []
    edgeColors = []
    for u, v in filmGraph.edges():
        if u not in positions or v not in positions:
            continue
        edgeSegments.append([positions[u], positions[v]])
        cu, cv = nodeToCommunity.get(u), nodeToCommunity.get(v)
        if cu == cv:
            edgeColors.append(to_rgba(commToColor[cu], alpha=0.12))
        else:
            edgeColors.append(to_rgba('#cccccc', alpha=0.03))
    
    if edgeSegments:
        lc = LineCollection(edgeSegments, colors=edgeColors, linewidths=0.3, zorder=1)
        ax.add_collection(lc)
    
    # draw nodes - separate western and non-western films
    for node in filmGraph.nodes():
        if node not in positions:
            continue
        
        x, y = positions[node]
        originalId = filmGraph.nodes[node].get('original_id')
        commId = nodeToCommunity.get(node, 0)
        color = commToColor[commId]
        pr = filmFilmRanks.get(originalId, 0)
        size = 20 + pr * 200
        
        # check if non-western
        if originalId in metaLookup:
            meta = metaLookup[originalId]
            isEU = meta.get('regions.eu', 0)
            isNA = meta.get('regions.na', 0)
            isWestern = (pd.notna(isEU) and int(isEU) == 1) or (pd.notna(isNA) and int(isNA) == 1)
            
            if not isWestern:
                # non-western: diamond with black edge
                ax.scatter(x, y, s=size*1.3, c=color, marker='D', alpha=0.9,
                          edgecolors='black', linewidths=0.8, zorder=3)
            else:
                # western: circle
                ax.scatter(x, y, s=size, c=color, marker='o', alpha=0.85,
                          edgecolors='white', linewidths=0.3, zorder=2)
        else:
            # no metadata: default circle
            ax.scatter(x, y, s=size, c=color, marker='o', alpha=0.85,
                      edgecolors='white', linewidths=0.3, zorder=2)
    
    # add community labels (high zorder to appear on top of all nodes)
    for commId in communities:
        commNodes = [n for n, c in nodeToCommunity.items() if c == commId and n in positions]
        if len(commNodes) < 10:
            continue
        
        cx = np.mean([positions[n][0] for n in commNodes])
        cy = np.mean([positions[n][1] for n in commNodes])
        
        label = labelCommunity(commNodes, filmGraph, filmMetadata, None, filmIdCol)
        
        ax.annotate(f"{label}\n(n={len(commNodes)})", (cx, cy), fontsize=8, ha='center', va='center',
                   fontweight='bold', color='#2c3e50', zorder=10,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                            edgecolor=commToColor[commId], alpha=0.95, linewidth=2))
    
    # legend for markers
    legendElements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
               markersize=10, label='Film (Western)'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='gray', 
               markeredgecolor='black', markersize=10, label='Film (Non-Western/Intl)'),
    ]
    ax.legend(handles=legendElements, loc='upper left', fontsize=10)
    
    ax.set_title('Film Network Only: FILMRANK\nFilms connected if they share festivals\nNode Size = FilmRank | Color = Community | Diamond = Non-Western',
                fontsize=14, fontweight='bold')
    ax.axis('off')
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(outputPath, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()


# EXPORT ALL FEATURES FOR DOWNSTREAM MODELING TASKS

def exportFeatures( filmFilmRanks,festivalFilmRanks, outputDir):
    """
    export all network features to CSV files.
    """
    outputPath = Path(outputDir)
    
    # film features
    allFilmIds = set(filmFilmRanks.keys())
    filmFeatures = pd.DataFrame({'film_id': list(allFilmIds)})
    filmFeatures['filmrank'] = filmFeatures['film_id'].map(filmFilmRanks).fillna(0)
    
    # festival features
    allFestIds = set(festivalFilmRanks.keys()) 
    festivalFeatures = pd.DataFrame({'festival_id': list(allFestIds)})
    festivalFeatures['filmrank'] = festivalFeatures['festival_id'].map(festivalFilmRanks).fillna(0)
    
    # save
    filmFeatures.to_csv(outputPath / 'film_network_features.csv', index=False)
    festivalFeatures.to_csv(outputPath / 'festival_network_features.csv', index=False)
    
    
    
    
    return filmFeatures, festivalFeatures


#------------------
#MAIN PIPELINE

def runPipeline(festivalRuns,filmMetadata=None,festivalMetadata=None, filmIdCol='unique.id', 
                festivalIdCol='festival.id', yearCol='event.year',  outputDir='./network_output',
                computeTemporal=True, yearsToCompute=None,generateVisualizations=True):
    """
    run the complete network analysis pipeline.
    """

    # handle festival metadata if provided as filepath
    if festivalMetadata is not None and isinstance(festivalMetadata, str):
        try:
            festivalMetadata = pd.read_csv(festivalMetadata, sep=';', low_memory=False)
        except:
            festivalMetadata = pd.read_csv(festivalMetadata, sep=',', low_memory=False)
    
    # data Validation
    requiredCols = [filmIdCol, festivalIdCol, yearCol]
    missingCols = [c for c in requiredCols if c not in festivalRuns.columns]
    if missingCols:
        altNames = {
            'unique.id': ['unique.id', 'film_id', 'film.id'],
            'festival.id': ['festival.id', 'fest.id'],
            'event.year': ['event.year', 'sample.year', 'year']
        }
        for missing in missingCols:
            for alt in altNames.get(missing, []):
                if alt in festivalRuns.columns:
                    if missing == filmIdCol:
                        filmIdCol = alt
                    elif missing == festivalIdCol:
                        festivalIdCol = alt
                    elif missing == yearCol:
                        yearCol = alt
                    break
    
    missingCols = [c for c in [filmIdCol, festivalIdCol, yearCol] if c not in festivalRuns.columns]
    if missingCols:
        raise ValueError(f"Missing required columns in festivalRuns: {missingCols}\n"
                        f"Available columns: {list(festivalRuns.columns)}")
    
    outputPath = Path(outputDir)
    outputPath.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    # xalculate total steps for tqdm
    totalSteps = 3  # Graph, FilmRank, Export
    if computeTemporal:
        totalSteps += 1
    if generateVisualizations:
        totalSteps += 1
    
    # main progress bar
    with tqdm(total=totalSteps, desc="Pipeline Progress", unit="step", 
              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar:
        
        # Build graph
        pbar.set_description("Building graph")
        G = createBipartiteGraph(festivalRuns, filmIdCol, festivalIdCol, yearCol)
        results['graph'] = G
        pbar.update(1)
        
        # FilmRank
        pbar.set_description("Computing FilmRank")
        filmPR, festPR = computeBipartiteFilmRank(G)
        results['film_filmranks'] = filmPR
        results['festival_filmranks'] = festPR
        pbar.update(1)
        
        
        # Temporal features
        if computeTemporal:
            pbar.set_description("Computing temporal features")
            tempPR = computeTemporalFilmRank(festivalRuns, filmIdCol, festivalIdCol, yearCol, yearsToCompute)
            results['temporal_filmrank'] = tempPR
            tempPR.to_csv(outputPath / 'temporal_filmrank.csv', index=False)
            pbar.update(1)
        
        # Visualizations
        if generateVisualizations:
            pbar.set_description("Generating visualizations")
            
            vizTasks = [
                ("Bipartite FilmRank", lambda: visualizeBipartiteFilmRank(
                    G, filmPR, festPR, filmMetadata, festivalMetadata, filmIdCol,
                    str(outputPath / 'viz_bipartite_pagerank.png'))),
                ("Films FilmRank", lambda: visualizeFilmsOnlyFilmRank(
                    G, filmPR, filmMetadata, filmIdCol,
                    str(outputPath / 'viz_films_pagerank.png')))
            ]
            
            for vizName, vizFunc in tqdm(vizTasks, desc="  Visualizations", leave=False, unit="viz"):
                vizFunc()
            
            pbar.update(1)
        
        # Step 6: Export features
        pbar.set_description("Exporting features")
        filmFeatures, festFeatures = exportFeatures(filmPR, festPR, str(outputPath))
        results['film_features'] = filmFeatures
        results['festival_features'] = festFeatures
        pbar.update(1)
        
        pbar.set_description("Complete")
    
    # Summary output
    print(f"Pipeline complete! Outputs saved to: {outputPath}")
    print(f"{len(filmFeatures)} films, {len(festFeatures)} festivals processed")
    if generateVisualizations:
        print(f"2 visualizations generated")
    
    return results



# MAIN
if __name__ == "__main__":
    festivalRuns = None
    filmMetadata = None
    
    # Run pipeline
    results = runPipeline(
        festivalRuns=festivalRuns,
        filmMetadata=filmMetadata,
        filmIdCol='unique.id',
        festivalIdCol='festival.id',
        yearCol='event.year',
        outputDir='./network_demo',
        computeTemporal=True,
        yearsToCompute=[2012, 2013, 2014]
    )