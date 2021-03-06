
/*
    Sequencial:
        real    0m10,550s
        user    0m0,015s
        sys     0m0,016s
*/
#include <iostream>
#include <vector>
#include <math.h>
#include <time.h>

using namespace std;

#define Train_Set_Size 20
#define PI 3.141592653589793238463
#define N 5
#define epsilon 0.05
#define epoch 70000

double c[N] = {};
double W[N] = {};
double V[N] = {};
double b = 0;

double sigmoid(double x) {
	return (1.0f / (1.0f + std::exp(-x)));
}

double f_theta(double x) {
	double result = b;
	for (int i = 0; i < N; i++) {
		result += V[i] * sigmoid(c[i] + W[i] * x);
	}
	return result;
}

void train(double x, double y) {
	int k;

    #pragma omp target map(tofrom:W[:N]) //map(to:c[:N]) map(to:W[:N])
    #pragma omp teams distribute parallel for simd reduction(+:W)
	for (int i = 0; i < N; i++) {
		W[i] = W[i] - epsilon * 2 * (f_theta(x) - y) * V[i] * x * (1 - sigmoid(c[i] + W[i] * x)) * sigmoid(c[i] + W[i] * x);
	}
   // #pragma omp target map(tofrom:V[:N]) map(to:c[:N]) map(to:W[:N])
   // #pragma omp teams distribute parallel for simd 
	for (int i = 0; i < N; i++) {
        k = epsilon * 2 * (f_theta(x) - y) * sigmoid(c[i] + W[i] * x);
		V[i] = V[i] - k;
	}
	b = b - epsilon * 2 * (f_theta(x) - y);
	//#pragma omp target map(tofrom:c[:N]) map(to:W[:N]) map(to:V[:N])
    //#pragma omp teams distribute parallel for simd private(k) reduction(+:V) schedule(dynamic,100)
    for (int i = 0; i < N; i++) {
        k = epsilon * 2 * (f_theta(x) - y) * V[i] * (1 - sigmoid(c[i] + W[i] * x)) * sigmoid(c[i] + W[i] * x);
		c[i] = c[i] - k;
	}
}

int main() {
	srand(time(NULL));
     
	for (int i = 0; i < N; i++) {
		W[i] = 2 * rand() / RAND_MAX -1;
		V[i] = 2 * rand() / RAND_MAX -1;
		c[i] = 2 * rand() / RAND_MAX -1;
	}
	vector<pair<double, double>> trainSet;
	trainSet.resize(Train_Set_Size);

	
	for (int i = 0; i < Train_Set_Size; i++) {
		trainSet[i] = make_pair(i * 2 * PI / Train_Set_Size, sin(i * 2 * PI / Train_Set_Size));
	}

	//#pragma omp parallel for num_threads(8)
	for (int j = 0; j < epoch; j++) {
		for (int i = 0; i < Train_Set_Size; i++) {
			train(trainSet[i].first, trainSet[i].second);
		}
		//std::cout << j << "\r";
	}

	//Plot the results
	vector<float> x;
	vector<float> y1, y2;

	for (int i = 0; i < 1000; i++) {
		x.push_back(i * 2 * PI / 1000);
		y1.push_back(sin(i * 2 * PI / 1000));
		y2.push_back(f_theta(i * 2 * PI / 1000));
	}


	/*//Neural Network Approximate f(x) = sin(x) -> Red Graph
	for (int k = 0; k < x.size(); k++) {
		printf("%f %f %f \n", x[k],y1[k],y2[k]);
	}
	printf("e\n");*/
	return 0;
}