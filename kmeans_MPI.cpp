// Implementation of the KMeans Algorithm with MPI
// Reference: http://mnemstudio.org/clustering-k-means-example-1.htm

#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <chrono>
#include <omp.h>
#include <mpi.h>

using namespace std;

class Point
{
private:
    int id_point, id_cluster;
    vector<double> values;
    int total_values;
    string name;

public:
    Point(int id_point, vector<double> &values, string name = "")
    {
        this->id_point = id_point;
        total_values = values.size();

        for (int i = 0; i < total_values; i++)
            this->values.push_back(values[i]);

        this->name = name;
        id_cluster = -1;
    }

    int getID()
    {
        return id_point;
    }

    void setCluster(int id_cluster)
    {
        this->id_cluster = id_cluster;
    }

    int getCluster()
    {
        return id_cluster;
    }

    double getValue(int index)
    {
        return values[index];
    }

    int getTotalValues()
    {
        return total_values;
    }

    void addValue(double value)
    {
        values.push_back(value);
    }

    string getName()
    {
        return name;
    }
};

class Cluster
{
private:
    int id_cluster;
    vector<double> central_values;
    vector<Point> points;

public:
    Cluster(int id_cluster, Point point)
    {
        this->id_cluster = id_cluster;

        int total_values = point.getTotalValues();

        for (int i = 0; i < total_values; i++)
            central_values.push_back(point.getValue(i));

        points.push_back(point);
    }

    void addPoint(Point point)
    {
        points.push_back(point);
    }

    bool removePoint(int id_point)
    {
        int total_points = points.size();

        for (int i = 0; i < total_points; i++)
        {
            if (points[i].getID() == id_point)
            {
                points.erase(points.begin() + i);
                return true;
            }
        }
        return false;
    }

    double getCentralValue(int index)
    {
        return central_values[index];
    }

    void setCentralValue(int index, double value)
    {
        central_values[index] = value;
    }

    Point getPoint(int index)
    {
        return points[index];
    }

    int getTotalPoints()
    {
        return points.size();
    }

    int getID()
    {
        return id_cluster;
    }

    void clearPoints()
    {
        points.clear();
    }
};

class KMeans
{
private:
    int K; // number of clusters
    int total_values, total_points, max_iterations;
    vector<Cluster> clusters;

    // return ID of nearest center (uses Euclidean distance)
    int getIDNearestCenter(Point &point)
    {
        double min_dist = numeric_limits<double>::max();
        int id_cluster_center = -1;

        for (int i = 0; i < K; i++)
        {
            double sum = 0.0;
            for (int j = 0; j < total_values; j++)
            {
                sum += pow(clusters[i].getCentralValue(j) - point.getValue(j), 2.0);
            }
            double dist = sqrt(sum);

            if (dist < min_dist)
            {
                min_dist = dist;
                id_cluster_center = i;
            }
        }

        return id_cluster_center;
    }

public:
    KMeans(int K, int total_points, int total_values, int max_iterations)
    {
        this->K = K;
        this->total_points = total_points;
        this->total_values = total_values;
        this->max_iterations = max_iterations;
    }

    void run(vector<Point> &all_points, int rank, int size)
    {
        if (K > total_points)
            return;

        int points_per_proc = total_points / size;
        int remainder = total_points % size;
        int start_index, end_index;

        if (rank < remainder)
        {
            start_index = rank * (points_per_proc + 1);
            end_index = start_index + points_per_proc + 1;
        }
        else
        {
            start_index = rank * points_per_proc + remainder;
            end_index = start_index + points_per_proc;
        }

        vector<Point> points(all_points.begin() + start_index, all_points.begin() + end_index);
        int local_total_points = points.size();

        // Initialize clusters (only on rank 0)
        if (rank == 0)
        {
            vector<int> prohibited_indexes;

            // Choose K distinct values for the centers of the clusters
            for (int i = 0; i < K; i++)
            {
                while (true)
                {
                    int index_point = rand() % total_points;

                    if (find(prohibited_indexes.begin(), prohibited_indexes.end(),
                             index_point) == prohibited_indexes.end())
                    {
                        prohibited_indexes.push_back(index_point);
                        all_points[index_point].setCluster(i);
                        Cluster cluster(i, all_points[index_point]);
                        clusters.push_back(cluster);
                        break;
                    }
                }
            }
        }

        // Broadcast initial clusters to all processes
        // Serialize cluster centers
        vector<double> cluster_centers(K * total_values);

        if (rank == 0)
        {
            for (int i = 0; i < K; i++)
            {
                for (int j = 0; j < total_values; j++)
                {
                    cluster_centers[i * total_values + j] = clusters[i].getCentralValue(j);
                }
            }
        }

        MPI_Bcast(cluster_centers.data(), K * total_values, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // Reconstruct clusters on other processes
        if (rank != 0)
        {
            clusters.clear();
            for (int i = 0; i < K; i++)
            {
                vector<double> center_values(total_values);
                for (int j = 0; j < total_values; j++)
                {
                    center_values[j] = cluster_centers[i * total_values + j];
                }
                Point dummy_point(-1, center_values);
                Cluster cluster(i, dummy_point);
                cluster.clearPoints(); // Remove the dummy point
                clusters.push_back(cluster);
            }
        }

        int iter = 1;
        while (true)
        {
            bool done = true;

            vector<int> new_clusters(local_total_points);

            // Assign points to the nearest cluster
#pragma omp parallel for schedule(static)
            for (int i = 0; i < local_total_points; i++)
            {
                int id_old_cluster = points[i].getCluster();
                int id_nearest_center = getIDNearestCenter(points[i]);

                new_clusters[i] = id_nearest_center;

                if (id_old_cluster != id_nearest_center)
                {
#pragma omp atomic write
                    done = false;
                }
            }

            // Gather the 'done' flag from all processes
            int global_done;
            MPI_Allreduce(&done, &global_done, 1, MPI_C_BOOL, MPI_LAND, MPI_COMM_WORLD);
            done = global_done;

            // Clear old points from clusters
            for (int i = 0; i < K; i++)
            {
                clusters[i].clearPoints();
            }

            // Assign points to clusters
            for (int i = 0; i < local_total_points; i++)
            {
                points[i].setCluster(new_clusters[i]);
                clusters[new_clusters[i]].addPoint(points[i]);
            }

            // Recalculate the center of each cluster
            vector<double> local_new_centers(K * total_values, 0.0);
            vector<int> local_counts(K, 0);

            for (int i = 0; i < K; i++)
            {
                int total_points_cluster = clusters[i].getTotalPoints();

                if (total_points_cluster > 0)
                {
                    local_counts[i] = total_points_cluster;

                    for (int p = 0; p < total_points_cluster; p++)
                    {
                        for (int j = 0; j < total_values; j++)
                        {
                            local_new_centers[i * total_values + j] += clusters[i].getPoint(p).getValue(j);
                        }
                    }
                }
            }

            // Reduce to get the global sums and counts
            vector<double> global_new_centers(K * total_values, 0.0);
            vector<int> global_counts(K, 0);

            MPI_Allreduce(local_new_centers.data(), global_new_centers.data(), K * total_values, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(local_counts.data(), global_counts.data(), K, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

            // Update cluster centers
            for (int i = 0; i < K; i++)
            {
                if (global_counts[i] > 0)
                {
                    for (int j = 0; j < total_values; j++)
                    {
                        clusters[i].setCentralValue(j, global_new_centers[i * total_values + j] / global_counts[i]);
                    }
                }
            }

            if (done == true || iter >= max_iterations)
            {
                if (rank == 0)
                    cout << "Break in iteration " << iter << "\n\n";
                break;
            }

            iter++;
        }

        // Optionally, gather results on rank 0 for final output
        // (This code is omitted for brevity)
    }
};

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    srand(time(NULL) + rank); // Different seed for each process

    // Start timing
    auto start = std::chrono::high_resolution_clock::now();

    // Default number of threads
    int num_threads = 1;

    if (argc > 1)
    {
        num_threads = atoi(argv[1]);
    }

    // Set the number of threads for parallelization
    omp_set_num_threads(num_threads);

    int total_points, total_values, K, max_iterations, has_name;

    vector<Point> all_points;

    if (rank == 0)
    {
        cin >> total_points >> total_values >> K >> max_iterations >> has_name;

        string point_name;

        for (int i = 0; i < total_points; i++)
        {
            vector<double> values;

            for (int j = 0; j < total_values; j++)
            {
                double value;
                cin >> value;
                values.push_back(value);
            }

            if (has_name)
            {
                cin >> point_name;
                Point p(i, values, point_name);
                all_points.push_back(p);
            }
            else
            {
                Point p(i, values);
                all_points.push_back(p);
            }
        }
    }

    // Broadcast parameters to all processes
    MPI_Bcast(&total_points, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&total_values, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&K, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&max_iterations, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&has_name, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Broadcast points to all processes
    // Serialize points
    int point_data_size = total_points * total_values;
    vector<double> point_data(point_data_size);

    if (rank == 0)
    {
        for (int i = 0; i < total_points; i++)
        {
            for (int j = 0; j < total_values; j++)
            {
                point_data[i * total_values + j] = all_points[i].getValue(j);
            }
        }
    }

    MPI_Bcast(point_data.data(), point_data_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Reconstruct all_points on other processes
    if (rank != 0)
    {
        all_points.clear();
        for (int i = 0; i < total_points; i++)
        {
            vector<double> values(total_values);
            for (int j = 0; j < total_values; j++)
            {
                values[j] = point_data[i * total_values + j];
            }
            Point p(i, values);
            all_points.push_back(p);
        }
    }

    KMeans kmeans(K, total_points, total_values, max_iterations);
    kmeans.run(all_points, rank, size);

    // Stop timing
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = finish - start;

    // Output execution time from rank 0
    if (rank == 0)
    {
        std::cout << "Execution time with " << size << " process(es) and " << num_threads << " thread(s): " << elapsed.count() << " seconds\n";
    }

    MPI_Finalize();

    return 0;
}
