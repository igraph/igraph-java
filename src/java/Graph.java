/* 
   IGraph library Java interface.
   Copyright (C) 2006-2012  Tamas Nepusz <ntamas@gmail.com>
   Pázmány Péter sétány 1/a, 1117 Budapest, Hungary
   
   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 2 of the License, or
   (at your option) any later version.
   
   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.
   
   You should have received a copy of the GNU General Public License
   along with this program; if not, write to the Free Software
   Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 
   02110-1301 USA 

*/

/*

ATTENTION: This is a highly experimental, proof-of-concept Java interface.
Its main purpose was to convince me that it can be done in finite time :)
The interface is highly incomplete, at the time of writing even some
essential functions (e.g. addEdges) are missing. Since I don't use Java
intensively, chances are that this interface gets finished only if there
is substantial demand for it and/or someone takes the time to send patches
or finish it completely.

*/

package net.sf.igraph;

public class Graph {
    private long handle=0; // handle to the underlying igraph_t object
  
    private Graph(long handle) { this.handle = handle; }
    private native void destroy();
  
    @Override protected void finalize() throws Throwable {
        if (handle != 0) destroy();
        super.finalize();
    }

	public static void test() {
		System.out.println("OK");
	}

    /* stimulus.py generated part starts here */
    public static native Graph Empty(long n, boolean directed);
    public native Graph addEdges(double[] edges, Object attr);
    public native Graph addVertices(long nv, Object attr);
    // deleteEdges: unknown input type EDGESET (needs JAVATYPE), skipping
    public native Graph deleteVertices(VertexSet vertices);
    public native long vcount();
    public native long ecount();
    public native double[] neighbors(long vid, NeighborMode mode);
    public native boolean isDirected();
    public native double[] degree(VertexSet vids, NeighborMode mode, boolean loops);
    // edge: calling convention unsupported yet
    // edges: unknown input type EDGESET (needs JAVATYPE), skipping
    public native long getEid(long from, long to, boolean directed, boolean error);
    public native double[] getEids(double[] pairs, boolean directed, boolean error);
    public native double[] incident(long vid, NeighborMode mode);
    public static native Graph Create(double[] edges, long n, boolean directed);
    // adjacency: unknown input type ADJACENCYMODE (needs JAVATYPE), skipping
    // weightedAdjacency: unknown input type ADJACENCYMODE (needs JAVATYPE), skipping
    public static native Graph Star(long n, StarMode mode, long center);
    public static native Graph Lattice(double[] dimvector, long nei, boolean directed, boolean mutual, boolean circular);
    public static native Graph Ring(long n, boolean directed, boolean mutual, boolean circular);
    // tree: unknown input type TREEMODE (needs JAVATYPE), skipping
    public static native Graph Full(long n, boolean directed, boolean loops);
    public static native Graph FullCitation(long n, boolean directed);
    public static native Graph Atlas(int number);
    public static native Graph ExtendedChordalRing(long nodes, double[][] W, boolean directed);
    public native Graph connectNeighborhood(long order, NeighborMode mode);
    public native Graph linegraph();
    public static native Graph DeBruijn(long m, long n);
    public static native Graph Kautz(long m, long n);
    public static native Graph Famous(String name);
    public static native Graph LcfVector(long n, double[] shifts, long repeats);
    // adjlist: unknown input type ADJLIST (needs JAVATYPE), skipping
    // fullBipartite: calling convention unsupported yet
    // barabasiGame: unknown input type BARABASI_ALGORITHM (needs JAVATYPE), skipping
    public static native Graph ErdosRenyiGameGnp(long n, double p, boolean directed, boolean loops);
    public static native Graph ErdosRenyiGameGnm(long n, double m, boolean directed, boolean loops);
    // degreeSequenceGame: unknown input type DEGSEQMODE (needs JAVATYPE), skipping
    public static native Graph GrowingRandomGame(long n, long m, boolean directed, boolean citation);
    public static native Graph BarabasiAgingGame(long nodes, long m, double[] outseq, boolean outpref, double pa_exp, double aging_exp, long aging_bin, double zero_deg_appeal, double zero_age_appeal, double deg_coef, double age_coef, boolean directed);
    public static native Graph RecentDegreeGame(long n, double power, long window, long m, double[] outseq, boolean outpref, double zero_appeal, boolean directed);
    public static native Graph RecentDegreeAgingGame(long nodes, long m, double[] outseq, boolean outpref, double pa_exp, double aging_exp, long aging_bin, long window, double zero_appeal, boolean directed);
    public static native Graph CallawayTraitsGame(long nodes, long types, long edges_per_step, double[] type_dist, double[][] pref_matrix, boolean directed);
    public static native Graph EstablishmentGame(long nodes, long types, long k, double[] type_dist, double[][] pref_matrix, boolean directed);
    public static native Graph GrgGame(long nodes, double radius, boolean torus, double[] x, double[] y);
    // preferenceGame: calling convention unsupported yet
    // asymmetricPreferenceGame: calling convention unsupported yet
    public native Graph rewireEdges(double prob, boolean loops);
    public static native Graph WattsStrogatzGame(long dim, long size, long nei, double p, boolean loops, boolean multiple);
    public static native Graph LastcitGame(long nodes, long edges_per_node, long agebins, double[] preference, boolean directed);
    public static native Graph CitedTypeGame(long nodes, double[] types, double[] pref, long edges_per_step, boolean directed);
    public static native Graph CitingCitedTypeGame(long nodes, double[] types, double[][] pref, long edges_per_step, boolean directed);
    public static native Graph ForestFireGame(long nodes, double fw_prob, double bw_factor, long ambs, boolean directed);
    public static native Graph SimpleInterconnectedIslandsGame(long islands_n, long islands_size, double islands_pin, long n_inter);
    public static native Graph StaticFitnessGame(long no_of_edges, double[] fitness_out, double[] fitness_in, boolean loops, boolean multiple);
    public static native Graph StaticPowerLawGame(long no_of_nodes, long no_of_edges, double exponent_out, double exponent_in, boolean loops, boolean multiple, boolean finite_size_correction);
    public static native Graph KRegularGame(long no_of_nodes, long k, boolean directed, boolean multiple);
    // sbmGame: unknown input type VECTOR_INT (needs JAVATYPE), skipping
    public static native Graph HsbmGame(long n, long m, double[] rho, double[][] C, double p);
    // hsbmListGame: unknown input type VECTOR_INT (needs JAVATYPE), skipping
    // correlatedGame: unknown input type VECTORM1_OR_0 (needs JAVATYPE), skipping
    // correlatedPairGame: calling convention unsupported yet
    public static native Graph DotProductGame(double[][] vecs, boolean directed);
    public static native double[][] SampleSphereSurface(long dim, long n, double radius, boolean positive);
    public static native double[][] SampleSphereVolume(long dim, long n, double radius, boolean positive);
    public static native double[][] SampleDirichlet(long n, double[] alpha);
    public native boolean areConnected(long v1, long v2);
    // diameter: calling convention unsupported yet
    // diameterDijkstra: calling convention unsupported yet
    // minimumSpanningTree: unknown input type EDGEWEIGHTS (needs JAVATYPE), skipping
    public native Graph minimumSpanningTreeUnweighted();
    public native Graph minimumSpanningTreePrim(double[] weights);
    // closeness: unknown input type EDGEWEIGHTS (needs JAVATYPE), skipping
    // closenessEstimate: unknown input type EDGEWEIGHTS (needs JAVATYPE), skipping
    public native double[][] shortestPaths(VertexSet from, VertexSet to, NeighborMode mode);
    // getShortestPaths: calling convention unsupported yet
    // getAllShortestPaths: calling convention unsupported yet
    // shortestPathsDijkstra: unknown input type EDGEWEIGHTS (needs JAVATYPE), skipping
    // getShortestPathsDijkstra: calling convention unsupported yet
    // getAllShortestPathsDijkstra: calling convention unsupported yet
    // shortestPathsBellmanFord: unknown input type EDGEWEIGHTS (needs JAVATYPE), skipping
    // shortestPathsJohnson: unknown input type EDGEWEIGHTS (needs JAVATYPE), skipping
    // getAllSimplePaths: unknown input type VERTEX (needs JAVATYPE), skipping
    // subcomponent: unknown input type VERTEX (needs JAVATYPE), skipping
    // betweenness: unknown input type EDGEWEIGHTS (needs JAVATYPE), skipping
    // betweennessEstimate: unknown input type EDGEWEIGHTS (needs JAVATYPE), skipping
    // edgeBetweenness: unknown input type EDGEWEIGHTS (needs JAVATYPE), skipping
    // edgeBetweennessEstimate: unknown input type EDGEWEIGHTS (needs JAVATYPE), skipping
    // pagerank: calling convention unsupported yet
    // personalizedPagerank: calling convention unsupported yet
    // rewire: unknown input type REWIRINGMODE (needs JAVATYPE), skipping
    // inducedSubgraph: unknown input type SUBGRAPH_IMPL (needs JAVATYPE), skipping
    public native Graph subgraph(VertexSet vids);
    // subgraphEdges: unknown input type EDGESET (needs JAVATYPE), skipping
    // averagePathLength: calling convention unsupported yet
    // pathLengthHist: calling convention unsupported yet
    // simplify: unknown input type EDGE_ATTRIBUTE_COMBINATION (needs JAVATYPE), skipping
    // transitivityUndirected: unknown input type TRANSITIVITYMODE (needs JAVATYPE), skipping
    // transitivityLocalUndirected: unknown input type TRANSITIVITYMODE (needs JAVATYPE), skipping
    // transitivityAvglocalUndirected: unknown input type TRANSITIVITYMODE (needs JAVATYPE), skipping
    // transitivityBarrat: unknown input type EDGEWEIGHTS (needs JAVATYPE), skipping
    // reciprocity: unknown input type RECIP (needs JAVATYPE), skipping
    public native double[] constraint(VertexSet vids, double[] weights);
    public native long maxdegree(VertexSet vids, NeighborMode mode, boolean loops);
    public native double density(boolean loops);
    public native double[] neighborhoodSize(VertexSet vids, long order, NeighborMode mode, long mindist);
    // neighborhood: unknown input type DEPS (needs JAVATYPE), skipping
    // neighborhoodGraphs: unknown return type GRAPHLIST, skipping
    public native double[] topologicalSorting(NeighborMode mode);
    // isLoop: unknown input type EDGESET (needs JAVATYPE), skipping
    public native boolean isDag();
    public native boolean isSimple();
    // isMultiple: unknown input type EDGESET (needs JAVATYPE), skipping
    public native boolean hasMultiple();
    // countMultiple: unknown input type EDGESET (needs JAVATYPE), skipping
    // girth: calling convention unsupported yet
    public native Graph addEdge(long from, long to);
    // eigenvectorCentrality: calling convention unsupported yet
    // hubScore: calling convention unsupported yet
    // authorityScore: calling convention unsupported yet
    // arpackRssolve: calling convention unsupported yet
    // arpackRnsolve: calling convention unsupported yet
    // arpackUnpackComplex: calling convention unsupported yet
    // unfoldTree: calling convention unsupported yet
    // isMutual: unknown input type EDGESET (needs JAVATYPE), skipping
    // maximumCardinalitySearch: calling convention unsupported yet
    // isChordal: calling convention unsupported yet
    // avgNearestNeighborDegree: calling convention unsupported yet
    // strength: unknown input type EDGEWEIGHTS (needs JAVATYPE), skipping
    public static native double Centralization(double[] scores, double theoretical_max, boolean normalized);
    // centralizationDegree: calling convention unsupported yet
    // centralizationDegreeTmax: unknown input type GRAPH_OR_0 (needs JAVATYPE), skipping
    // centralizationBetweenness: calling convention unsupported yet
    // centralizationBetweennessTmax: unknown input type GRAPH_OR_0 (needs JAVATYPE), skipping
    // centralizationCloseness: calling convention unsupported yet
    // centralizationClosenessTmax: unknown input type GRAPH_OR_0 (needs JAVATYPE), skipping
    // centralizationEigenvectorCentrality: calling convention unsupported yet
    // centralizationEigenvectorCentralityTmax: unknown input type GRAPH_OR_0 (needs JAVATYPE), skipping
    // assortativityNominal: unknown input type VECTORM1 (needs JAVATYPE), skipping
    public native double assortativity(double[] types1, double[] types2, boolean directed);
    public native double assortativityDegree(boolean directed);
    // contractVertices: unknown input type VECTORM1 (needs JAVATYPE), skipping
    // eccentricity: unknown return type VERTEXINDEX, skipping
    public native double radius(NeighborMode mode);
    // diversity: unknown input type EDGEWEIGHTS (needs JAVATYPE), skipping
    // randomWalk: unknown input type VERTEX (needs JAVATYPE), skipping
    public static native boolean IsDegreeSequence(double[] out_deg, double[] in_deg);
    public static native boolean IsGraphicalDegreeSequence(double[] out_deg, double[] in_deg);
    // bfs: calling convention unsupported yet
    // dfs: calling convention unsupported yet
    // bipartiteProjectionSize: calling convention unsupported yet
    // bipartiteProjection: calling convention unsupported yet
    // createBipartite: unknown input type VECTOR_BOOL (needs JAVATYPE), skipping
    // incidence: calling convention unsupported yet
    // getIncidence: calling convention unsupported yet
    // isBipartite: calling convention unsupported yet
    // bipartiteGameGnp: calling convention unsupported yet
    // bipartiteGameGnm: calling convention unsupported yet
    // laplacian: calling convention unsupported yet
    // clusters: calling convention unsupported yet
    public native boolean isConnected(Connectedness mode);
    // decompose: unknown input type LONGINT (needs JAVATYPE), skipping
    public native VertexSet articulationPoints();
    // biconnectedComponents: calling convention unsupported yet
    // cliques: unknown return type VERTEXSETLIST, skipping
    // largestCliques: unknown return type VERTEXSETLIST, skipping
    // maximalCliques: unknown return type VERTEXSETLIST, skipping
    public native long maximalCliquesCount(long min_size, long max_size);
    // maximalCliquesFile: unknown input type OUTFILE (needs JAVATYPE), skipping
    public native long cliqueNumber();
    // independentVertexSets: unknown return type VERTEXSETLIST, skipping
    // largestIndependentVertexSets: unknown return type VERTEXSETLIST, skipping
    // maximalIndependentVertexSets: unknown return type VERTEXSETLIST, skipping
    public native long independenceNumber();
    public native double[][] layoutRandom();
    public native double[][] layoutCircle(VertexSet order);
    // layoutStar: unknown input type VERTEX (needs JAVATYPE), skipping
    public native double[][] layoutGrid(long width);
    public native double[][] layoutGrid3d(long width, long height);
    // layoutFruchtermanReingold: unknown input type LAYOUT_GRID (needs JAVATYPE), skipping
    // layoutKamadaKawai: unknown input type EDGEWEIGHTS (needs JAVATYPE), skipping
    public native double[][] layoutLgl(long maxiter, double maxdelta, double area, double coolexp, double repulserad, double cellsize, long root);
    public native double[][] layoutReingoldTilford(NeighborMode mode, double[] roots);
    public native double[][] layoutReingoldTilfordCircular(NeighborMode mode, double[] roots);
    public native double[][] layoutRandom3d();
    public native double[][] layoutSphere();
    // layoutFruchtermanReingold3d: unknown input type EDGEWEIGHTS (needs JAVATYPE), skipping
    // layoutKamadaKawai3d: unknown input type EDGEWEIGHTS (needs JAVATYPE), skipping
    // layoutGraphopt: unknown input type NULL (needs JAVATYPE), skipping
    // layoutDrl: unknown input type DRL_OPTIONS (needs JAVATYPE), skipping
    // layoutDrl3d: unknown input type DRL_OPTIONS (needs JAVATYPE), skipping
    // layoutMergeDla: unknown input type GRAPHLIST (needs JAVATYPE), skipping
    // layoutSugiyama: calling convention unsupported yet
    // layoutMds: unknown input type MATRIX_OR_0 (needs JAVATYPE), skipping
    // layoutBipartite: unknown input type BIPARTITE_TYPES (needs JAVATYPE), skipping
    public native double[][] layoutGem(boolean use_seed, long maxiter, double temp_max, double temp_min, double temp_init);
    public native double[][] layoutDavidsonHarel(boolean use_seed, long maxiter, long fineiter, double cool_fact, double weight_node_dist, double weight_border, double weight_edge_lengths, double weight_edge_crossings, double weight_node_edge_dist);
    public native double[][] cocitation(VertexSet vids);
    public native double[][] bibcoupling(VertexSet vids);
    public native double[][] similarityJaccard(VertexSet vids, NeighborMode mode, boolean loops);
    public native double[][] similarityDice(VertexSet vids, NeighborMode mode, boolean loops);
    public native double[][] similarityInverseLogWeighted(VertexSet vids, NeighborMode mode);
    // compareCommunities: unknown input type COMMCMP (needs JAVATYPE), skipping
    // communitySpinglass: calling convention unsupported yet
    // communitySpinglassSingle: calling convention unsupported yet
    // communityWalktrap: calling convention unsupported yet
    // communityEdgeBetweenness: calling convention unsupported yet
    // communityEbGetMerges: calling convention unsupported yet
    // communityFastgreedy: calling convention unsupported yet
    // communityToMembership: calling convention unsupported yet
    // leCommunityToMembership: calling convention unsupported yet
    public static native double Modularity();
    // modularityMatrix: unknown input type EDGEWEIGHTS (needs JAVATYPE), skipping
    // reindexMembership: calling convention unsupported yet
    // communityLeadingEigenvector: calling convention unsupported yet
    // communityFluidCommunities: calling convention unsupported yet
    // communityLabelPropagation: calling convention unsupported yet
    // communityMultilevel: calling convention unsupported yet
    // communityOptimalModularity: calling convention unsupported yet
    // communityLeiden: calling convention unsupported yet
    // splitJoinDistance: calling convention unsupported yet
    // hrgFit: unknown return type HRG, skipping
    // hrgGame: unknown input type HRG (needs JAVATYPE), skipping
    // hrgDendrogram: unknown input type HRG (needs JAVATYPE), skipping
    // hrgConsensus: calling convention unsupported yet
    // hrgPredict: calling convention unsupported yet
    // hrgCreate: unknown return type HRG, skipping
    // communityInfomap: calling convention unsupported yet
    // graphlets: calling convention unsupported yet
    // graphletsCandidateBasis: calling convention unsupported yet
    // graphletsProject: unknown input type EDGEWEIGHTS (needs JAVATYPE), skipping
    // getAdjacency: unknown input type GETADJACENCY (needs JAVATYPE), skipping
    public native double[] getEdgelist(boolean bycol);
    // toDirected: unknown input type TODIRECTED (needs JAVATYPE), skipping
    // toUndirected: unknown input type TOUNDIRECTED (needs JAVATYPE), skipping
    public native double[][] getStochastic(boolean column_wise);
    // getStochasticSparsemat: unknown return type SPARSEMATPTR, skipping
    // readGraphEdgelist: unknown input type INFILE (needs JAVATYPE), skipping
    // readGraphNcol: unknown input type INFILE (needs JAVATYPE), skipping
    // readGraphLgl: unknown input type INFILE (needs JAVATYPE), skipping
    // readGraphPajek: unknown input type INFILE (needs JAVATYPE), skipping
    // readGraphGraphml: unknown input type INFILE (needs JAVATYPE), skipping
    // readGraphDimacs: calling convention unsupported yet
    // readGraphGraphdb: unknown input type INFILE (needs JAVATYPE), skipping
    // readGraphGml: unknown input type INFILE (needs JAVATYPE), skipping
    // readGraphDl: unknown input type INFILE (needs JAVATYPE), skipping
    // writeGraphEdgelist: unknown input type OUTFILE (needs JAVATYPE), skipping
    // writeGraphNcol: unknown input type OUTFILE (needs JAVATYPE), skipping
    // writeGraphLgl: unknown input type OUTFILE (needs JAVATYPE), skipping
    // writeGraphLeda: unknown input type OUTFILE (needs JAVATYPE), skipping
    // writeGraphGraphml: unknown input type OUTFILE (needs JAVATYPE), skipping
    // writeGraphPajek: unknown input type OUTFILE (needs JAVATYPE), skipping
    // writeGraphDimacs: unknown input type OUTFILE (needs JAVATYPE), skipping
    // writeGraphGml: unknown input type OUTFILE (needs JAVATYPE), skipping
    // writeGraphDot: unknown input type OUTFILE (needs JAVATYPE), skipping
    public native double[] motifsRandesu(int size, double[] cut_prob);
    public native long motifsRandesuEstimate(int size, double[] cut_prob, long sample_size, double[] sample);
    public native long motifsRandesuNo(int size, double[] cut_prob);
    // dyadCensus: calling convention unsupported yet
    public native double[] triadCensus();
    public native double[] adjacentTriangles(VertexSet vids);
    // localScan0: unknown input type EDGEWEIGHTS (needs JAVATYPE), skipping
    // localScan0Them: unknown input type EDGEWEIGHTS (needs JAVATYPE), skipping
    // localScan1Ecount: unknown input type EDGEWEIGHTS (needs JAVATYPE), skipping
    // localScan1EcountThem: unknown input type EDGEWEIGHTS (needs JAVATYPE), skipping
    // localScanKEcount: unknown input type EDGEWEIGHTS (needs JAVATYPE), skipping
    // localScanKEcountThem: unknown input type EDGEWEIGHTS (needs JAVATYPE), skipping
    // localScanNeighborhoodEcount: unknown input type EDGEWEIGHTS (needs JAVATYPE), skipping
    // listTriangles: unknown return type VERTEXSET_INT, skipping
    // disjointUnion: calling convention unsupported yet
    // disjointUnionMany: calling convention unsupported yet
    public native Graph union(Graph right);
    // unionMany: unknown input type GRAPHLIST (needs JAVATYPE), skipping
    public native Graph intersection(Graph right);
    // intersectionMany: unknown input type GRAPHLIST (needs JAVATYPE), skipping
    public native Graph difference(Graph sub);
    public native Graph complementer(boolean loops);
    public native Graph compose(Graph g2);
    // maxflow: calling convention unsupported yet
    // maxflowValue: calling convention unsupported yet
    public native double mincutValue(double[] capacity);
    // stMincutValue: unknown input type VERTEX (needs JAVATYPE), skipping
    // mincut: calling convention unsupported yet
    // stVertexConnectivity: unknown input type VERTEX (needs JAVATYPE), skipping
    public native long vertexConnectivity(boolean checks);
    // stEdgeConnectivity: unknown input type VERTEX (needs JAVATYPE), skipping
    public native long edgeConnectivity(boolean checks);
    // edgeDisjointPaths: unknown input type VERTEX (needs JAVATYPE), skipping
    // vertexDisjointPaths: unknown input type VERTEX (needs JAVATYPE), skipping
    public native long adhesion(boolean checks);
    public native long cohesion(boolean checks);
    // dominatorTree: calling convention unsupported yet
    // allStCuts: calling convention unsupported yet
    // allStMincuts: calling convention unsupported yet
    public native boolean isSeparator(VertexSet candidate);
    public native boolean isMinimalSeparator(VertexSet candidate);
    // allMinimalStSeparators: unknown return type VERTEXSETLIST, skipping
    // minimumSizeSeparators: unknown return type VERTEXSETLIST, skipping
    // cohesiveBlocks: calling convention unsupported yet
    public native double[] coreness(NeighborMode mode);
    public native long isoclass();
    public native boolean isomorphic(Graph graph2);
    public native long isoclassSubgraph(double[] vids);
    public static native Graph IsoclassCreate(long size, long number, boolean directed);
    // isomorphicVf2: calling convention unsupported yet
    // countIsomorphismsVf2: unknown input type VERTEX_COLOR (needs JAVATYPE), skipping
    // getIsomorphismsVf2: unknown input type VERTEX_COLOR (needs JAVATYPE), skipping
    // subisomorphicVf2: calling convention unsupported yet
    // countSubisomorphismsVf2: unknown input type VERTEX_COLOR (needs JAVATYPE), skipping
    // getSubisomorphismsVf2: unknown input type VERTEX_COLOR (needs JAVATYPE), skipping
    public native boolean isomorphic34(Graph graph2);
    // canonicalPermutation: calling convention unsupported yet
    // permuteVertices: unknown input type VECTORM1 (needs JAVATYPE), skipping
    // isomorphicBliss: calling convention unsupported yet
    // automorphisms: unknown input type NULL (needs JAVATYPE), skipping
    // subisomorphicLad: calling convention unsupported yet
    // scgGrouping: unknown input type SCGMAT (needs JAVATYPE), skipping
    // scgSemiprojectors: calling convention unsupported yet
    // scgNormEps: unknown input type VECTORM1 (needs JAVATYPE), skipping
    // scgAdjacency: calling convention unsupported yet
    // scgStochastic: calling convention unsupported yet
    // scgLaplacian: calling convention unsupported yet
    // isMatching: unknown input type BIPARTITE_TYPES_OR_0 (needs JAVATYPE), skipping
    // isMaximalMatching: unknown input type BIPARTITE_TYPES_OR_0 (needs JAVATYPE), skipping
    // maximumBipartiteMatching: calling convention unsupported yet
    // adjacencySpectralEmbedding: calling convention unsupported yet
    // laplacianSpectralEmbedding: calling convention unsupported yet
    // eigenAdjacency: calling convention unsupported yet
    // powerLawFit: unknown return type PLFIT, skipping
    // sir: unknown return type SIRLIST, skipping
    public static native double[] RunningMean(double[] data, long binwidth);
    public static native double[] RandomSample(long l, long h, long length);
    // convexHull: calling convention unsupported yet
    public static native long DimSelect(double[] sv);
    // isEulerian: unknown input type BOOL (needs JAVATYPE), skipping
    // eulerianPath: unknown return type ERROR, skipping
    // eulerianCycle: unknown return type ERROR, skipping
    // convergenceDegree: calling convention unsupported yet
    /* stimulus.py generated part ends here */
  
    static {
        System.loadLibrary("igraph-java-wrapper");
    }
}
