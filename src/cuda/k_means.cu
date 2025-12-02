#include "ORBextractor.h"
#include "k_means.h"

#define TDB 1024
#define N_ITER 1

__device__ inline float distance(ORB_SLAM3::GpuPoint x1, int2 x2)
{
    const float diffx = (float)x2.x - x1.x;
    const float diffy = (float)x2.y - x1.y;
	return (diffx * diffx) + (diffy * diffy);
}

/*
 * CUDA kernel that assigns each datapoint to the closest centroid
 * in a K-means clustering algorithm. Each thread processes one datapoint.
 *
 * Parameters:
 * - d_datapoints_: pointer to the array of datapoints (ORB_SLAM3::GpuPoint)
 * - final_points_: pointer to the output array of final clustered points
 * - d_centroids_: pointer to the array of centroids (int2)
 * - N_: pointer to an array containing the number of datapoints per pyramid level
 * - K_: pointer to an array containing the number of clusters (centroids) per pyramid level
 * - maxLevel: number of pyramid levels
 * - points_offset: offset for datapoints per level
 * - cluster_offset: offset for centroids per level
 *
 * Behavior/notes:
 * - Each thread computes the distance from its assigned datapoint to all centroids,
 *   finds the closest centroid, and assigns the cluster ID to the datapoint.
*/
__global__ void kMeansClusterAssignment(ORB_SLAM3::GpuPoint *d_datapoints_, ORB_SLAM3::GpuPoint *final_points_, 
	int2 *d_centroids_, uint *N_, int *K_, int maxLevel, int points_offset, int cluster_offset)
{
	//get idx for this datapoint
	const int level = blockIdx.y;
	const int idx = blockIdx.x*blockDim.x + threadIdx.x;

	if (level >= maxLevel) return;

	const uint N = N_[level];

	//bounds check
	if (idx >= N) return;

	const int K = K_[level];

	ORB_SLAM3::GpuPoint *d_datapoints = &(d_datapoints_[points_offset*level]);
	ORB_SLAM3::GpuPoint *final_points = &(final_points_[points_offset*level]);
	final_points[idx].clust_assn = -1;
	int2 *d_centroids = &(d_centroids_[cluster_offset*level]);

	//find the closest centroid to this datapoint
	float min_dist = INFINITY;
	int closest_centroid = 0;

	for(int c=0; c<K; c++)
	{
		float dist = distance(d_datapoints[idx],d_centroids[c]);

		if(dist < min_dist)
		{
			min_dist = dist;
			closest_centroid = c;
		}
	}

	//assign closest cluster id for this datapoint/thread
	d_datapoints[idx].clust_assn = closest_centroid;
}

/*
 * CUDA kernel that computes the maximum score for each cluster
 * in a K-means clustering algorithm. Each thread processes one datapoint.
 *
 * Parameters:
 * - d_datapoints_: pointer to the array of datapoints (ORB_SLAM3::GpuPoint)
 * - N_: pointer to an array containing the number of datapoints per pyramid level
 * - k_scores_: pointer to an array to store the maximum score per cluster
 * - maxLevel: number of pyramid levels
 * - points_offset: offset for datapoints per level
 * - cluster_offset: offset for clusters per level
 *
 * Behavior/notes:
 * - Each thread reads its assigned datapoint's cluster assignment and score,
 *   and updates the maximum score for that cluster using atomic operations.
*/
__global__ void get_max_score(ORB_SLAM3::GpuPoint *d_datapoints_, uint *N_, int *k_scores_, int maxLevel, int points_offset, int cluster_offset)
{
	const int level = blockIdx.y;
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (level >= maxLevel) return;

	const uint N = N_[level];

	if (idx >= N) return;

	ORB_SLAM3::GpuPoint *d_datapoints = &(d_datapoints_[points_offset*level]);
	int *k_scores = &(k_scores_[cluster_offset*level]);

	const int k = d_datapoints[idx].clust_assn;
	const int score = d_datapoints[idx].score;

	atomicMax(&(k_scores[k]), score);
}

/*
 * CUDA kernel that selects the datapoint with the maximum score
 * for each cluster in a K-means clustering algorithm. Each thread processes one datapoint.
 *
 * Parameters:
 * - d_datapoints_: pointer to the array of datapoints (ORB_SLAM3::GpuPoint)
 * - N_: pointer to an array containing the number of datapoints per pyramid level
 * - k_scores_: pointer to an array containing the maximum score per cluster
 * - final_points_: pointer to the output array of final clustered points
 * - maxLevel: number of pyramid levels
 * - points_offset: offset for datapoints per level
 * - cluster_offset: offset for clusters per level
 *
 * Behavior/notes:
 * - Each thread checks if its assigned datapoint's score matches the maximum score
 *   for its cluster, and if so, writes it to the final points array.
*/
__global__ void get_max_points(ORB_SLAM3::GpuPoint *d_datapoints_, uint *N_, int *k_scores_, ORB_SLAM3::GpuPoint *final_points_, 
	int maxLevel, int points_offset, int cluster_offset)
{
	const int level = blockIdx.y;
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (level >= maxLevel) return;

	const uint N = N_[level];

	if (idx >= N) return;

	ORB_SLAM3::GpuPoint *d_datapoints = &(d_datapoints_[points_offset*level]);
	ORB_SLAM3::GpuPoint *final_points = &(final_points_[points_offset*level]);
	int *k_scores = &(k_scores_[cluster_offset*level]);

	const int k = d_datapoints[idx].clust_assn;
	const int score = d_datapoints[idx].score;

	const int max_score = k_scores[k];

	if (max_score == score) {
		final_points[k] = d_datapoints[idx];
	} else {
		d_datapoints[idx].clust_assn = -1;
	}

}

/*
 * CUDA kernel that copies final clustered points from the temporary
 * final points array back to the original datapoints array.
 *
 * Parameters:
 * - d_datapoints_: pointer to the array of datapoints (ORB_SLAM3::GpuPoint)
 * - K_: pointer to an array containing the number of clusters (centroids) per pyramid level
 * - N_: pointer to an array containing the number of datapoints per pyramid level
 * - final_points_: pointer to the array of final clustered points
 * - maxLevel: number of pyramid levels
 * - points_offset: offset for datapoints per level
 *
 * Behavior/notes:
 * - Each thread checks if its assigned final point has a valid cluster assignment,
 *   and if so, copies it back to the original datapoints array.
*/

__global__ void copy_points(ORB_SLAM3::GpuPoint *d_datapoints_, int *K_, uint *N_, ORB_SLAM3::GpuPoint *final_points_, int maxLevel, int points_offset)
{
	const int level = blockIdx.y;
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (level >= maxLevel) return;

	const int K = K_[level];

	if (idx >= K) return;

	ORB_SLAM3::GpuPoint *d_datapoints = &(d_datapoints_[points_offset*level]);
	ORB_SLAM3::GpuPoint *final_points = &(final_points_[points_offset*level]);

	if (final_points[idx].clust_assn != -1)
		d_datapoints[idx] = final_points[idx];

}
/*
 * CUDA kernel that assigns any remaining unassigned clusters
 * to the closest datapoint in a K-means clustering algorithm.
 * Each thread processes one cluster.
 *
 * Parameters:
 * - d_datapoints_: pointer to the array of datapoints (ORB_SLAM3::GpuPoint)
 * - final_points_: pointer to the output array of final clustered points
 * - k_scores_: pointer to an array containing the maximum score per cluster
 * - d_centroids_: pointer to the array of centroids (int2)
 * - N_: pointer to an array containing the number of datapoints per pyramid level
 * - K_: pointer to an array containing the number of clusters (centroids) per pyramid level
 * - maxLevel: number of pyramid levels
 * - points_offset: offset for datapoints per level
 * - cluster_offset: offset for centroids per level
 *
 * Behavior/notes:
 * - Each thread checks if its assigned cluster has no assigned datapoint,
 *   finds the closest datapoint, and assigns it to the final points array.
*/
__global__ void last_cluster_assign(ORB_SLAM3::GpuPoint *d_datapoints_, ORB_SLAM3::GpuPoint *final_points_, int *k_scores_, 
	int2 *d_centroids_, uint *N_, int *K_, int maxLevel, int points_offset, int cluster_offset)
{
	//get idx for this datapoint
	const int level = blockIdx.y;
	const int idx = blockIdx.x*blockDim.x + threadIdx.x;

	if (level >= maxLevel) return;

	const uint N = N_[level];
	const int K = K_[level];

	ORB_SLAM3::GpuPoint *final_points = &(final_points_[points_offset*level]);

	//bounds check
	if (idx >= K || final_points[idx].clust_assn != -1) return;

	ORB_SLAM3::GpuPoint *d_datapoints = &(d_datapoints_[points_offset*level]);
	int2 *d_centroids = &(d_centroids_[cluster_offset*level]);

	float min_dist = INFINITY;
	int closest_point = 0;

	for(int i=0; i<N; i++)
	{
		if (d_datapoints[i].clust_assn != -1) continue;

		float dist = distance(d_datapoints[i],d_centroids[idx]);
		
		if(dist < min_dist)
		{
			min_dist = dist;
			closest_point = i;
		}
	}

	final_points[idx] = d_datapoints[closest_point];
}


void filter_points(ORB_SLAM3::GpuPoint *d_datapoints, ORB_SLAM3::GpuPoint *final_points_buffer, int2 *d_centroids, int *d_clust_sizes, uint *N, int *K, int2 *initial_centroids, int *mnFeatrues, int maxLevel, int points_offset, int cluster_offset, cudaStream_t stream) {
	dim3 dg( ceil( (float)points_offset/TDB ), maxLevel );
	dim3 dg2( ceil( (float)cluster_offset/TDB ), maxLevel );
    dim3 db( TDB );

	cudaMemcpyAsync(d_centroids, initial_centroids, sizeof(int2)*maxLevel*cluster_offset, cudaMemcpyDeviceToDevice, stream);
	cudaMemsetAsync(d_clust_sizes,0,cluster_offset*maxLevel*sizeof(int), stream);

	kMeansClusterAssignment<<<dg, db, 0, stream>>>(d_datapoints, final_points_buffer, d_centroids, N, K, maxLevel, points_offset, cluster_offset);
	get_max_score<<<dg, db, 0, stream>>>(d_datapoints, N, d_clust_sizes, maxLevel, points_offset, cluster_offset);
	get_max_points<<<dg, db, 0, stream>>>(d_datapoints, N, d_clust_sizes, final_points_buffer, maxLevel, points_offset, cluster_offset);
	last_cluster_assign<<<dg2, db, 0, stream>>>(d_datapoints, final_points_buffer, d_clust_sizes, d_centroids, N, K, maxLevel, points_offset, cluster_offset);
	copy_points<<<dg2, db, 0, stream>>>(d_datapoints, K, N, final_points_buffer, maxLevel, points_offset);
}