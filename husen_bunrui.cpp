// husen_bunrui.cpp : コンソール アプリケーションのエントリ ポイントを定義します。
//

#include "stdafx.h"
#define _USE_MATH_DEFINES
#include <iostream>
#include <cstdio>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "opencv2\core\core.hpp"
#include "opencv2\highgui\highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"


using namespace std;
using namespace cv;

struct cbpt {
	FILE *f;
	Mat m;
};

void mouse_callback(int event, int x, int y, int flags, void *p);
Point2f fmean2i(Point* points, int len, int axis);
float norm(Point2f p);
int naiseki(Point p1, Point p2);
int gaiseki(Point p1, Point p2);
double tan_kakudo(float x, float y);
double tentosen_dict(Point q1, Point p1, Point p2);
double fusen_dict(vector<Point>points1, vector<Point>points2);
vector<int>neibor(int id, double eps, vector<Point2f>data_center, int*classed);
vector<int>neibor_fusen(int id, double eps, vector<vector<Point>>data, int*classed);
vector<int>neibor_kyori_kakudo(int id, vector<double> epss, vector<vector<Point>>data, int*classed);
vector<int>core(int id, double eps, vector<Point2f>data_center, int*classed);
vector<int>core_fusen(int id, double eps, vector<vector<Point>>data, int*classed);
vector<int>core_kyori_kakudo(int id, vector<double> epss, vector<vector<Point>>data, int*classed);
vector<vector<int>> dbscan(vector<Point2f>data_center, double eps);
vector<vector<int>> dbscan_fusen(vector<vector<Point>>data, double eps);
vector<vector<int>> dbscan_kyori_kakudo(vector<vector<Point>>data, vector<double>epss);
vector<vector<int>>calc_cross(vector<vector<int>>A, vector<vector<int>>C);
double calc_entropy(vector<vector<int>>X, int N);
double calc_purity(vector<vector<int>>X, int N);
double calc_F(vector<vector<int>> X, int N);
double calc_P(vector<vector<int>> X, int N);
double calc_R(vector<vector<int>> X, int N);
int makeHusenCsv(string file_name, string csv_file_name);
int loopMakeHusenCsv();
int checkAsData(vector<vector<vector<vector<int>>>>*all_A_,
	vector<vector<vector<Point2f>>>*all_husen_center_data_,
	vector<vector<vector<vector<Point>>>>*all_husen_points_data_);

double calcF(vector<vector<int>>, vector<vector<int>>, int);
double calcFp(vector<vector<int>>, vector<vector<int>>, int);
double calcP(vector<vector<int>>, vector<vector<int>>, int);
double calcR(vector<vector<int>>, vector<vector<int>>, int);
int mitukeru(int, vector<vector<int>>);
template <class X> X modo(X* start, int size);


int main()
{
	//loopMakeHusenCsv();
	vector<vector<vector<vector<int>>>>all_A;
	vector<vector<vector<Point2f>>>all_husen_center_data;
	vector<vector<vector<vector<Point>>>>all_husen_points_data;
	checkAsData(&all_A, &all_husen_center_data, &all_husen_points_data);

	int i, j;
	const int ngroup = 4;
	const int nsample = 10;
	FILE *fp;
	FILE * output_file;

	////pair 3parameter
	//int ccc = 0;
	//fopen_s(&fp, "3paraplot0619tikainodake.csv", "w");
	//for (i = 0; i < ngroup; i++) {
	//	for (j = 0; j < nsample; j++) {
	//		vector<vector<int>> A = all_A[i][j];
	//		vector<vector<int>> C;
	//		vector<Point2f> husen_center = all_husen_center_data[i][j];
	//		vector<vector<Point>>husen_points = all_husen_points_data[i][j];

	//		char file_name[100];
	//		sprintf_s(file_name, 100, "./gazou_g%d_%d.jpg", i + 1, j + 1);
	//		Mat im = imread(file_name, 1);
	//		Mat resized(int(im.rows*0.25), int(im.cols*0.25), im.type());
	//		cv::resize(im, resized, resized.size());

	//		int k, l, m, n;
	//		for (k = 0; k < A.size(); k++) {
	//			for (l = 0; l < A[k].size(); l++) {
	//				double min_dict = -1;
	//				int min_arg;
	//				double min_arg_info[3];
	//				for (m = 0; m < A.size(); m++) {
	//					for (n = 0; n < A[m].size(); n++) {
	//						int id1;
	//						id1 = A[k][l];
	//						int id2;
	//						id2 = A[m][n];
	//						//printf("%d,%d\n", id1, id2);
	//						if (id1 != id2) {
	//							
	//							Point2f pt1;
	//							Point2f pt2;
	//							double costhe;
	//							double tan1;
	//							double tan2;
	//							double kakudo1;
	//							double kakudo2;
	//							double theta;
	//							//printf("%d,%d\n", husen_center[id1].x, husen_center[id1].y);
	//							double r = norm(husen_center[id1] - husen_center[id2]);
	//							double cosome;
	//							double omega;
	//							double d1, d2;

	//							/*
	//							d1 = norm(husen_points[id][0] - husen_points[id][1]);
	//							d2 = norm(husen_points[id][1] - husen_points[id][2]);
	//							if (d1 > d2) {
	//							pt1 = husen_points[id][0] - husen_points[id][1];
	//							}
	//							else {
	//							pt1 = husen_points[id][1] - husen_points[id][2];
	//							}
	//							//*/

	//							pt1 = Point2f((husen_points[id1][1] - husen_points[id1][0]).x, (husen_points[id1][1] - husen_points[id1][0]).y);
	//							pt2 = husen_center[id2] - husen_center[id1];
	//							//printf("%lf,%lf\n", pt1.x, pt1.y);
	//							//printf("%lf,%lf\n", pt2.x, pt2.y);
	//							kakudo1 = tan_kakudo(pt1.x, pt1.y);
	//							kakudo2 = tan_kakudo(pt2.x, pt2.y);
	//							//costhe = naiseki(pt1, pt2) / (norm(pt1)*norm(pt2));
	//							theta = kakudo2 - kakudo1;

	//							if (theta < 0) {
	//								theta += 2 * M_PI;
	//							}

	//							/*
	//							d1 = norm(husen_points[id][0] - husen_points[id][1]);
	//							d2 = norm(husen_points[id][1] - husen_points[id][2]);
	//							if (d1 > d2) {
	//							pt1 = husen_points[id][0] - husen_points[id][1];
	//							}
	//							else {
	//							pt1 = husen_points[id][1] - husen_points[id][2];
	//							}
	//							id = id2;
	//							d1 = norm(husen_points[id][0] - husen_points[id][1]);
	//							d2 = norm(husen_points[id][1] - husen_points[id][2]);
	//							if (d1 > d2) {
	//							pt2 = husen_points[id][0] - husen_points[id][1];
	//							}
	//							else {
	//							pt2 = husen_points[id][1] - husen_points[id][2];
	//							}
	//							*/
	//							//printf("%d,%d\n", id1, id2);
	//							pt1 = Point2f((husen_points[id1][1] - husen_points[id1][0]).x, (husen_points[id1][1] - husen_points[id1][0]).y);
	//							pt2 = Point2f((husen_points[id2][1] - husen_points[id2][0]).x, (husen_points[id2][1] - husen_points[id2][0]).y);
	//							//printf("%lf,%lf\n", pt1.x, pt1.y);
	//							//printf("%lf,%lf\n", pt2.x, pt2.y);
	//							kakudo1 = tan_kakudo(pt1.x, pt1.y);
	//							kakudo2 = tan_kakudo(pt2.x, pt2.y);

	//							//cosome = naiseki(pt1, pt2) / (norm(pt1)*norm(pt2));
	//							omega = kakudo2 - kakudo1;
	//							if (omega < 0) {
	//								omega += 2 * M_PI;
	//							}

	//							if (k == m) {
	//								printf("%d,%d,%d,%d,%d,0,%lf,%lf,%lf\n", ccc, i, j, id1, id2, r, theta, omega);
	//								fprintf(fp, "%d,%d,%d,%d,0,%lf,%lf,%lf\n", i, j, id1, id2, r, theta, omega);
	//								if ( min_dict < 0 ||min_dict > r) {
	//									min_dict = r;
	//									min_arg = id2;
	//									min_arg_info[0] = r;
	//									min_arg_info[1] = theta;
	//									min_arg_info[2] = omega;
	//								}
	//							}
	//							else {
	//								printf("%d, %d,%d,%d,%d,1,%lf,%lf,%lf\n", ccc, i, j, id1, id2, r, theta, omega);
	//								fprintf(fp, "%d,%d,%d,%d,1,%lf,%lf,%lf\n", i, j, id1, id2, r, theta, omega);
	//							}
	//							ccc++;
	//						}

	//					}
	//				}
	//				if (min_dict > 0) {
	//					printf("%d,%d,%d,%d,%d,2,%lf,%lf,%lf\n", ccc, i, j, A[k][l], min_arg, min_arg_info[0], min_arg_info[1], min_arg_info[2]);
	//					fprintf(fp, "%d,%d,%d,%d,2,%lf,%lf,%lf\n", i, j, A[k][l], min_arg, min_arg_info[0], min_arg_info[1], min_arg_info[2]);
	//				}
	//				
	//			}
	//		}
	//	}
	//}
	//fclose(fp);

	/*グループごとに3:1を4回*/
double min_eps = 70;
double max_eps = 80;
double deps = 0.2;
double eps;

	

vector<double> vec_good_eps;
int group;
	for (group = 0; group < ngroup; group++){
		double max_sum_f = 0;
		double good_eps = 0;
		for (eps = min_eps; eps < max_eps; eps += deps) {
			double sum_f = 0;
			for (i = 0; i < ngroup; i++) {
				if (i != group) {
					for (j = 0; j < nsample; j++) {
						vector<vector<int>> A = all_A[i][j];
						vector<vector<int>> C;
						vector<Point2f> husen_center = all_husen_center_data[i][j];
						C = dbscan(husen_center, eps);
						double dF = calcFp(A, C, husen_center.size());
						sum_f += dF;
					}
				}
			}
			if (max_sum_f < sum_f) {
				max_sum_f = sum_f;
				good_eps = eps;
			}
		}
		vec_good_eps.push_back(good_eps);
	}
	for (i = 0; i < ngroup; i++) {
		printf("eps,%lf\n", vec_good_eps[i]);
	}

	/* atteruno count 
	vector<vector<int>> rAs(ngroup);
	for (i = 0; i < ngroup; i++) {
		vector<int>rA(all_A[i][0].size(), 0);
		for (j = 0; j < nsample; j++) {
			vector<vector<int>> A = all_A[i][j];
			vector<vector<int>> C;
			vector<Point2f> husen_center = all_husen_center_data[i][j];
			C = dbscan(husen_center, vec_good_eps[i]);
			vector<vector<int>> X = calc_cross(A, C);
			int k, l;
			for (l = 0; l < X[0].size(); l++) {
				vector<int>maxvecX;
				int maxX = 0;
				for (k = 0; k < X.size(); k++) {
					if (maxX < X[k][l]) {
						maxX = X[k][l];
						maxvecX = X[k];
					}
				}
				string prin("");
				for (k = 0; k < maxvecX.size(); k++) {
					char mimi[100];
					sprintf_s(mimi, 100, "%d,", maxvecX[k]);
					prin += mimi;
				}
				prin.pop_back();
				prin += "\n";
				rA[l] += maxX;
				printf("%s", &prin[0]);
			}
		}
		rAs[i] = rA;
		for (j = 0; j < rA.size(); j++) {
			//printf("%d/%d,", rA[j], all_A[i][0][j].size()*nsample);
		}
		printf("\n");
	}
	//*/


fopen_s(&output_file, "husen_31_prF_dbscan_center0620.csv", "w");

double minF = 1.0;
int min_group;
int min_sample;
vector<Point2f> min_husen_center;
vector<vector<int>> minC;
for (i = 0; i < ngroup; i++) {
	for (j = 0; j < nsample; j++) {
		vector<vector<int>> A = all_A[i][j];
		vector<vector<int>> C;
		vector<Point2f> husen_center = all_husen_center_data[i][j];
		vector<vector<Point>>husen_points = all_husen_points_data[i][j];
		C = dbscan(husen_center, vec_good_eps[i]);

		double F = calcFp(A, C, husen_center.size());
		double r = calcR(A, C, husen_center.size());
		double p = calcP(A, C, husen_center.size());
		//F = 2 * r*p / (r + p);

		fprintf_s(output_file, ",,,,,3,%d,%d,%d,%lf,%lf,%lf\n", i + 1, j + 1, C.size(), p, r, F);
		printf(",,,,,3,%d,%d,%d,%lf,%lf,%lf\n", i + 1, j + 1, C.size(), p, r, F);
		char file_name[100];

		//*
		sprintf_s(file_name, 100, "./gazou_g%d_%d.jpg", i + 1, j + 1);
		Mat im = imread(file_name, 1);
		Mat resized(int(im.rows*0.25), int(im.cols*0.25), im.type());
		resize(im, resized, resized.size());
		int k, l;
		for (k = 0; k < C.size(); k++) {
			for (l = 0; l < C[k].size(); l++) {
				char str[10];
				sprintf_s(str, 10, "%d", k);
				Point h_center((int)husen_center[C[k][l]].x, (int)husen_center[C[k][l]].y);
				putText(resized, str, h_center, CV_FONT_HERSHEY_SIMPLEX, 1.5, Scalar(0, 0, 255));
				//cv::putText(resized, "OpenCV", cv::Point(50, 50), CV_FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(0, 0, 200), 2, CV_AA);
			}
		}
		sprintf_s(file_name, 100, "./bunrui0616_g%d_%d.jpg", i + 1, j + 1);
		imwrite(file_name, resized);
		//*/

		if (F < minF) {
			minF = F;
			min_group = i;
			min_sample = j;
			min_husen_center = husen_center;
			minC = C;
		}
	}
}
fclose(output_file);


	fopen_s(&output_file, "husen_31_prF_dbscan_kyori_kakudo0620.csv", "w");
	//*31 eps sagasu
	vector<vector<double>> vec_good_epss;
	//vec_good_eps.clear();
	double max_sum_f = 0;

	vector<double> goodf;

	for (group = 0; group < ngroup; group++) {
		max_sum_f = 0;
		double min_eps[4][3] = { { 65, 60, 60 } ,{ 65, 60, 60 },{ 65, 60,60 } ,{ 65, 60, 60 } };
		double max_eps[4][3] = { { 75, 85, 70 },{ 85, 85, 70 },{ 85, 85, 70 },{ 85, 85, 70 } };
		double deps[4] = { 0.5, 0.5, 0.5, 0.5 };
		vector<double> eps(3);
		vector<double> good_eps(3);
		for (eps[0] = min_eps[group][0]; eps[0] < max_eps[group][0]; eps[0] += deps[group]) {
			for (eps[1] = min_eps[group][1]; eps[1] < max_eps[group][1]; eps[1] += deps[group]) {
				for (eps[2] = min_eps[group][2]; eps[2] < max_eps[group][2]; eps[2] += deps[group]) {
					double sum_f = 0;
					for (i = 0; i < ngroup; i++) {
						if (i != group) {
							for (j = 0; j < nsample; j++) {
								vector<vector<Point>>husen_points = all_husen_points_data[i][j];
								vector<vector<int>> A = all_A[i][j];
								vector<vector<int>> C;
								C = dbscan_kyori_kakudo(husen_points, eps);
								sum_f += calcFp(A, C, husen_points.size());
							}
						}
					}
					if (max_sum_f < sum_f) {
						max_sum_f = sum_f;
						good_eps[0] = eps[0];
						good_eps[1] = eps[1];
						good_eps[2] = eps[2];
						printf("%d:F=%lf,eps,%lf,%lf,%lf\n", group, max_sum_f, good_eps[0], good_eps[1], good_eps[2]);
					}
				}
			}

		}
		goodf.push_back(max_sum_f);
		vec_good_epss.push_back(good_eps);
	}
	for (i = 0; i < ngroup; i++) {
		printf("group%d:F=%lf,eps,%lf,%lf,%lf\n", i, goodf[i], vec_good_epss[i][0], vec_good_epss[i][1], vec_good_epss[i][2]);
	}

	printf("\n%lf,%lf,%lf, %lf\n", vec_good_epss[0], vec_good_epss[1], vec_good_epss[2], max_sum_f / ngroup / nsample);
	for (i = 0; i<ngroup; i++) {
		for (j = 0; j < nsample; j++) {
			vector<vector<Point>>husen_points = all_husen_points_data[i][j];
			vector<Point2f> husen_center = all_husen_center_data[i][j];
			vector<vector<int>> A = all_A[i][j];
			vector<vector<int>> C;
			C = dbscan_kyori_kakudo(husen_points, vec_good_epss[i]);
			double F = calcFp(A, C, husen_points.size());
			double r = calcR(A, C, husen_points.size());
			double p = calcP(A, C, husen_points.size());
			F = 2 * r*p / (r + p);

			//*
			fprintf_s(output_file, ",,,,,4,%d,%d,%d,%lf,%lf,%lf\n", i + 1, j + 1, C.size(), p, r, F);
			printf(",,,,,4,%d,%d,%d,%lf,%lf,%lf\n", i + 1, j + 1, C.size(), p, r, F);
			char file_name[100];
			sprintf_s(file_name, 100, "./gazou_g%d_%d.jpg", i + 1, j + 1);
			Mat im = imread(file_name, 1);
			Mat resized(int(im.rows*0.25), int(im.cols*0.25), im.type());
			resize(im, resized, resized.size());
			int k, l;
			for (k = 0; k < C.size(); k++) {
				for (l = 0; l < C[k].size(); l++) {
					char str[10];
					sprintf_s(str, 10, "%d", k);
					Point h_center((int)husen_center[C[k][l]].x, (int)husen_center[C[k][l]].y);
					putText(resized, str, h_center, CV_FONT_HERSHEY_SIMPLEX, 1.5, Scalar(0, 0, 255));
					//cv::putText(resized, "OpenCV", cv::Point(50, 50), CV_FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(0, 0, 200), 2, CV_AA);
				}
			}
			sprintf_s(file_name, 100, "./bunrui_kyori_kakudo0620_g%d_%d.jpg", i + 1, j + 1);
			imwrite(file_name, resized);
			//*/
		}
	}
	fclose(output_file);
    return 0;
}

void mouse_callback(int event, int x, int y, int flags, void *p) {
	FILE * fp;
	static int count;

	fp = ((struct cbpt*)p)->f;
	Mat resized = ((struct cbpt*)p)->m;
	if (event == CV_EVENT_LBUTTONDOWN) {
		printf("%d, %d,%d\n", count, x, y);
		circle(resized, cv::Point(x, y), 1, Scalar(0, 0, 255), 1);
		imshow("mi", resized);
		fprintf(fp, "%d,%d\n", x, y);
		count++;
	}
}
Point2f fmean2i(Point* points, int len, int axis) {
	int i;
	float mean_x = 0;
	float mean_y = 0;

	if (axis == 0) {
		for (i = 0; i < len; i++) {
			mean_x += points[i].x;
			mean_y += points[i].y;
		}
	}
	return Point2f(mean_x / len, mean_y / len);
}
float norm(Point2f p) {
	float dist = sqrt(pow(p.x, 2) + pow(p.y, 2));
	return dist;
}
int naiseki(Point p1, Point p2) {
	return p1.x*p2.x + p2.y*p1.y;
}
int gaiseki(Point p1, Point p2) {
	return p1.x*p2.y - p1.y*p2.x;
}
double tan_kakudo(float x, float y)
{
	double kakudo;
	if (x != 0) {
		double tan = y / x;
		if (x > 0) {
			kakudo = atan(tan);
		}
		else {
			kakudo = atan(tan) + M_PI;
		}
		if (kakudo < 0) {
			kakudo += 2 * M_PI;
		}
	}

	else {
		if (y > 0) {
			kakudo = M_PI / 2;
		}
		else {
			kakudo = M_PI * 3 / 2.0;
		}
	}
	return kakudo;
}
double tentosen_dict(Point q1, Point p1, Point p2) {
	int nseki = naiseki(q1 - p1, p2 - p1);
	int nseki2 = naiseki(p2 - p1, p2 - p1);

	if (nseki < 0) {
		return (double)norm(p1 - q1);
	}
	else if (nseki <nseki2) {
		double d2 = nseki / sqrt((double)nseki2);
		return sqrt(naiseki(q1 - p1, q1 - p1) - pow(d2, 2));
	}
	else {
		return (double)norm(p2 - q1);
	}

}
double fusen_dict(vector<Point>points1, vector<Point>points2) {
	int i, j;
	double min_dict = -1;
	double dict;
	for (i = 0; i < points1.size(); i++) {
		for (j = 0; j < points2.size(); j++) {
			int gseki1 = gaiseki(points1[i] - points2[j], points2[(j + 1) % 4] - points2[j]);
			int gseki2 = gaiseki(points1[(i + 1) % 4] - points2[j], points2[(j + 1) % 4] - points2[j]);
			int gseki3 = gaiseki(points2[j] - points1[i], points1[(i + 1) % 4] - points1[i]);
			int gseki4 = gaiseki(points2[(j + 1) % 4] - points1[i], points1[(i + 1) % 4] - points1[i]);
			if (gseki1 / 1.0 * gseki2 / 1.0 < 0) {
				if (gseki3 / 1.0 * gseki4 / 1.0 < 0) {
					return 0;
				}
			}
			dict = tentosen_dict(points1[i], points2[j], points2[(j + 1) % 4]);
			if (min_dict == -1 || dict < min_dict) {
				min_dict = dict;
			}
			dict = tentosen_dict(points2[i], points1[j], points1[(j + 1) % 4]);
			if (min_dict == -1 || dict < min_dict) {
				min_dict = dict;
			}
		}
	}
	return min_dict;
}

vector<int>neibor(int id, double eps, vector<Point2f>data_center, int*classed) {
	int i = 0;
	vector<int>neibor_id;
	for (i = 0; i < data_center.size(); i++) {
		if (i != id) {
			if (classed[i] == false) {
				if (norm(data_center[id] - data_center[i]) <= eps) {
					neibor_id.push_back(i);
				}
			}
		}
	}
	return neibor_id;
}
vector<int>neibor_fusen(int id, double eps, vector<vector<Point>>data, int*classed) {
	int i = 0;
	vector<int>neibor_id;
	for (i = 0; i < data.size(); i++) {
		if (i != id) {
			if (classed[i] == false) {
				if (fusen_dict(data[id], data[i]) <= eps) {
					neibor_id.push_back(i);
				}
			}
		}
	}
	return neibor_id;
}
vector<int> neibor_kyori_kakudo(int id, vector<double> epss, vector<vector<Point>> data, int * classed)
{
	int i = 0;
	vector<int>neibor_id;
	for (i = 0; i < data.size(); i++) {
		if (i != id) {
			if (classed[i] == false) {
				double dict = norm(fmean2i(&data[i][0], 4, 0) - fmean2i(&data[id][0], 4, 0));
				double theta = atan(0.75);
				double cosome;
				double omega;
				double max_dict;
				Point pt1 = data[id][1] - data[id][0];
				Point2f pt2 = fmean2i(&data[i][0], 4, 0) - fmean2i(&data[id][0], 4, 0);
				cosome = (pt1.x*pt2.x + pt1.y*pt2.y) / (norm(pt1)*norm(pt2));
				omega = acos(cosome);
				if (omega > M_PI_2) {
					omega = M_PI - omega;
				}
				if (omega < theta) {
					max_dict = epss[0] + (omega / theta)*(epss[1] - epss[0]);
				}
				else {
					max_dict = epss[1] + (omega - theta) / (M_PI / 2.0 - theta)*(epss[2] - epss[1]);
				}
				if (dict <= max_dict) {
					neibor_id.push_back(i);
				}
			}
		}
	}
	return neibor_id;
}
vector<int>core(int id, double eps, vector<Point2f>data_center, int*classed) {

	vector<int>core_id;
	vector<int>neibor_id = neibor(id, eps, data_center, classed);
	core_id.push_back(id);
	classed[id] = true;
	if (neibor_id.size() == 0) {
		return core_id;
	}
	else {
		int i;
		for (i = 0; i < neibor_id.size(); i++) {
			classed[neibor_id[i]] = true;
		}
		for (i = 0; i < neibor_id.size(); i++) {
			vector<int> temp = core(neibor_id[i], eps, data_center, classed);
			int j;
			for (j = 0; j < temp.size(); j++) {
				core_id.push_back(temp[j]);
			}
		}
	}
	return core_id;
}
vector<int>core_fusen(int id, double eps, vector<vector<Point>>data, int*classed) {

	vector<int>core_id;
	vector<int>neibor_id = neibor_fusen(id, eps, data, classed);
	core_id.push_back(id);
	classed[id] = true;
	if (neibor_id.size() == 0) {
		return core_id;
	}
	else {
		int i;
		for (i = 0; i < neibor_id.size(); i++) {
			classed[neibor_id[i]] = true;
		}
		for (i = 0; i < neibor_id.size(); i++) {
			vector<int> temp = core_fusen(neibor_id[i], eps, data, classed);
			int j;
			for (j = 0; j < temp.size(); j++) {
				core_id.push_back(temp[j]);
			}
		}
	}
	return core_id;
}
vector<int>core_kyori_kakudo(int id, vector<double> epss, vector<vector<Point>>data, int*classed) {
	vector<int>core_id;
	vector<int>neibor_id = neibor_kyori_kakudo(id, epss, data, classed);
	core_id.push_back(id);
	classed[id] = true;
	if (neibor_id.size() == 0) {
		return core_id;
	}
	else {
		int i;
		for (i = 0; i < neibor_id.size(); i++) {
			classed[neibor_id[i]] = true;
		}
		for (i = 0; i < neibor_id.size(); i++) {
			vector<int> temp = core_kyori_kakudo(neibor_id[i], epss, data, classed);
			int j;
			for (j = 0; j < temp.size(); j++) {
				core_id.push_back(temp[j]);
			}
		}
	}
	return core_id;
}
vector<vector<int>> dbscan(vector<Point2f>data_center, double eps) {
	int * classed = (int *)malloc(sizeof(int) * data_center.size());
	int i;
	for (i = 0; i < data_center.size(); i++) {
		classed[i] = false;
	}
	int count_classed = 0;
	vector<vector<int>> C;

	int cls = 0;

	while (1) {
		C.push_back(vector<int>());
		for (i = 0; i < data_center.size(); i++) {
			if (classed[i] == false) {
				break;
			}
		}
		//C[cls].push_back(i);
		//classed[i] = true;
		//count_classed++;
		vector<int> core_id = core(i, eps, data_center, classed);
		int j;
		for (j = 0; j < core_id.size(); j++) {
			C[cls].push_back(core_id[j]);
			//classed[core_id[j]] = true;
			count_classed++;
		}
		cls++;
		if (count_classed >= data_center.size()) {
			break;
		}
	}
	free(classed);

	return C;

}
vector<vector<int>> dbscan_fusen(vector<vector<Point>>data, double eps) {
	int * classed = (int *)malloc(sizeof(int) * data.size());
	int i;
	for (i = 0; i < data.size(); i++) {
		classed[i] = false;
	}
	int count_classed = 0;
	vector<vector<int>> C;

	int cls = 0;

	while (1) {
		C.push_back(vector<int>());
		for (i = 0; i < data.size(); i++) {
			if (classed[i] == false) {
				break;
			}
		}
		//C[cls].push_back(i);
		//classed[i] = true;
		//count_classed++;
		vector<int> core_id = core_fusen(i, eps, data, classed);
		int j;
		for (j = 0; j < core_id.size(); j++) {
			C[cls].push_back(core_id[j]);
			//classed[core_id[j]] = true;
			count_classed++;
		}
		cls++;
		if (count_classed >= data.size()) {
			break;
		}
	}
	return C;
}
vector<vector<int>> dbscan_kyori_kakudo(vector<vector<Point>>data, vector<double>epss) {
	int * classed = (int *)malloc(sizeof(int) * data.size());
	int i;
	for (i = 0; i < data.size(); i++) {
		classed[i] = false;
	}
	int count_classed = 0;
	vector<vector<int>> C;

	int cls = 0;

	while (1) {
		C.push_back(vector<int>());
		for (i = 0; i < data.size(); i++) {
			if (classed[i] == false) {
				break;
			}
		}
		//C[cls].push_back(i);
		//classed[i] = true;
		//count_classed++;
		vector<int> core_id = core_kyori_kakudo(i, epss, data, classed);
		int j;
		for (j = 0; j < core_id.size(); j++) {
			C[cls].push_back(core_id[j]);
			//classed[core_id[j]] = true;
			count_classed++;
		}
		cls++;
		if (count_classed >= data.size()) {
			break;
		}
	}
	free(classed);
	return C;
}

vector<vector<int>>calc_cross(vector<vector<int>>A, vector<vector<int>>C) {
	vector<vector<int>> X;
	int i;
	for (i = 0; i < C.size(); i++) {
		X.push_back(vector<int>(A.size()));
		int j;
		for (j = 0; j < A.size(); j++) {
			int count = 0;
			int k, l;
			for (k = 0; k < C[i].size(); k++) {
				for (l = 0; l < A[j].size(); l++) {
					if (C[i][k] == A[j][l]) {
						count++;
					}
				}
			}
			X[i][j] = count;
		}
	}
	return X;
}


double calc_entropy(vector<vector<int>>X, int N) {
	double entropy = 0;
	int i;
	for (i = 0; i < X.size(); i++) {
		int Ci = 0;
		int j;
		double ent = 0;
		for (j = 0; j < X[i].size(); j++) {
			Ci += X[i][j];
		}
		if (Ci == 0) {
			break;
		}
		for (j = 0; j < X[i].size(); j++) {
			double p = 1.0 * X[i][j] / Ci;
			if (p == 0) {
				ent += 0;
				continue;
			}
			ent += -p * log(p);
		}
		entropy += 1.0 * Ci / N * ent;
	}
	return entropy;
}

double calc_purity(vector<vector<int>>X, int N) {
	double purity = 0;
	int i;
	for (i = 0; i < X.size(); i++) {
		int j;
		int max = 0;
		for (j = 0; j < X[i].size(); j++) {
			if (max < X[i][j]) {
				max = X[i][j];
			}
		}
		purity += max;
	}
	purity /= 1.0 * N;
	return purity;
}

double calc_P(vector<vector<int>> X, int N) {
	vector<int>A_num(X[0].size(), 0);
	vector<int>C_num(X.size(), 0);
	int i, j;
	for (i = 0; i < X.size(); i++) {
		for (j = 0; j < X[i].size(); j++) {
			A_num[j] += X[i][j];
			C_num[i] += X[i][j];
		}
	}

	double Pti = 0;
	for (j = 0; j < A_num.size(); j++) {
		double maxP = 0;
		double maxF = 0;
		for (i = 0; i < C_num.size(); i++) {
			double Pij = 1.0* X[i][j] / C_num[i];
			double Rij = 1.0 * X[i][j] / A_num[j];
			double F = 2.0*Pij*Rij / (Pij + Rij);
			//if (maxF < F) {
			if (maxP < Pij) {
				maxF = F;
				maxP = Pij;
			}
		}
		Pti += A_num[j] / (1.0*N) * maxP;
	}
	return Pti;
}
double calc_F(vector<vector<int>> X, int N) {
	vector<int>A_num(X[0].size(), 0);
	vector<int>C_num(X.size(), 0);
	int i, j;
	for (i = 0; i < X.size(); i++) {
		for (j = 0; j < X[i].size(); j++) {
			A_num[j] += X[i][j];
			C_num[i] += X[i][j];
		}
	}

	double Fti = 0;
	for (j = 0; j < A_num.size(); j++) {
		double maxF = 0;
		double maxR = 0;
		double maxP = 0;
		for (i = 0; i < C_num.size(); i++) {
			double Pij = 1.0* X[i][j] / C_num[i];
			double Rij = 1.0 * X[i][j] / A_num[j];
			double F = 2.0*Pij*Rij / (Pij + Rij);
			if (maxF < F) {
				maxF = F;
				maxR = Rij;
				maxP = Pij;
			}
		}
		Fti += A_num[j] / (1.0*N) * maxF;
	}
	return Fti;
}

double calc_R(vector<vector<int>> X, int N) {
	vector<int>A_num(X[0].size(), 0);
	vector<int>C_num(X.size(), 0);
	int i, j;
	for (i = 0; i < X.size(); i++) {
		for (j = 0; j < X[i].size(); j++) {
			A_num[j] += X[i][j];
			C_num[i] += X[i][j];
		}
	}

	double Rti = 0;
	for (j = 0; j < A_num.size(); j++) {
		double maxR = 0;
		double maxF = 0;
		for (i = 0; i < C_num.size(); i++) {
			double Pij = 1.0* X[i][j] / C_num[i];
			double Rij = 1.0 * X[i][j] / A_num[j];
			double F = 2.0*Pij*Rij / (Pij + Rij);
			//if (maxF < F) {
			if (maxR < Rij) {
				//maxF = F;
				//maxR = Rij;
				maxR = Rij;
			}
		}
		Rti += maxR / N * A_num[j];
	}
	return Rti;
}

int makeHusenCsv(string file_name, string csv_file_name) {
	FILE *fp;


	fopen_s(&fp, &file_name[0], "r");
	if (fp == NULL) {
		return 0;
	}
	fclose(fp);

	Mat mi = imread(file_name);
	Mat resized(int(mi.rows*0.25), int(mi.cols*0.25), mi.type());
	resize(mi, resized, resized.size());
	namedWindow("mi");
	imshow("mi", resized);
	fopen_s(&fp, &csv_file_name[0], "w");
	struct cbpt cb1;
	cb1.f = fp;
	cb1.m = resized;

	setMouseCallback("mi", mouse_callback, &cb1);
	while (1) {
		char c = waitKey(100);
		//printf("%d", c);
		if (c == 27) {
			break;
		}
		if (c == 'a') {
			fprintf(fp, "\n");
			printf("enter");
		}
		if (c == 'r') {
			fclose(fp);
			fopen_s(&fp, &csv_file_name[0], "w");
			resize(mi, resized, resized.size());
			printf("reset\n");
		}
	}
	fclose(fp);
	imwrite(file_name + "ten.png", resized);
	return 0;
}

int loopMakeHusenCsv()
{
	//loop makeHusenCsv
	int i;
	for (i = 18; i < 19; i++) {
		char file_name[100];
		sprintf_s(file_name, 100, "gazou_g%d_%d.JPG", i / 10 + 1, i % 10 + 1);
		makeHusenCsv(string(file_name), string(file_name) + "0530ul.csv");
	}
	return 0;
}

int checkAsData(vector<vector<vector<vector<int>>>>*all_A_,
	vector<vector<vector<Point2f>>>*all_husen_center_data_,
	vector<vector<vector<vector<Point>>>>*all_husen_points_data_)
{
	int num_group = 4;
	int num_sample = 10;

	int i, j;
	//*
	vector<vector<vector<vector<vector<Point>>>>>all_category_data(num_group);
	vector<vector<vector<vector<int>>>> all_A(num_group);
	vector<vector<vector<Point>>>all_husen_center_data(num_group);
	vector<vector<vector<vector<Point>>>>all_husen_points_data(num_group);
	//*/

	for (i = 0; i < num_group; i++) {
		vector<vector<vector<vector<Point>>>> team_category_data(num_sample);
		vector<vector<vector<int>>> team_A(num_sample);
		vector<vector<Point2f>>team_husen_center_data(num_sample);
		vector<vector<vector<Point>>>team_husen_points_data(num_sample);

		for (j = 0; j < num_sample; j++) {
			vector<vector<vector<Point>>> categories_points;
			vector<vector<int>> A;
			vector<Point2f>husen_center_data;
			vector<vector<Point>>husen_points_data;

			FILE *fp_husen_csv;
			char husen_csv_filename[100];
			sprintf_s(husen_csv_filename, 100, "gazou_g%d_%d.JPG0530ul.csv", i + 1, j + 1);
			fopen_s(&fp_husen_csv, husen_csv_filename, "r");
			if (fp_husen_csv == NULL) {
				break;
			}

			char pts_str[100];

			char * p;
			vector<vector<Point>> similar_husens;
			vector<Point> husen_points;

			int husen_count = 0;

			while (1) {
				if (fgets(pts_str, 100, fp_husen_csv) == NULL) {
					if (similar_husens.size() != 0) {
						categories_points.push_back(similar_husens);
						int k;
						vector<int> similar_id;
						for (k = 0; k < similar_husens.size(); k++) {
							similar_id.push_back(husen_count);
							husen_count += 1;
						}
						A.push_back(similar_id);

						similar_husens.clear();
						similar_id.clear();
					}
					team_category_data[j] = categories_points;
					team_husen_center_data[j] = husen_center_data;
					team_husen_points_data[j] = husen_points_data;
					similar_husens = vector<vector<Point>>();
					team_A[j] = A;
					A.clear();
					fclose(fp_husen_csv);
					break;
				}
				p = strchr(pts_str, ',');

				if (p == NULL) {
					categories_points.push_back(similar_husens);

					int k;
					vector<int> similar_id;
					for (k = 0; k < similar_husens.size(); k++) {
						similar_id.push_back(husen_count);
						husen_count += 1;
					}
					A.push_back(similar_id);

					//similar_husens = vector<vector<Point>>();
					similar_husens.clear();
					similar_id.clear();
				}
				else {
					char pts_str1[100];
					char pts_str2[100];
					int l = strlen(pts_str);
					int i;
					for (i = 0; i < p - pts_str; i++) {
						pts_str1[i] = pts_str[i];
					}
					pts_str1[i] = 0;
					for (i = 0; i < l - (p - pts_str) - 1; i++) {
						pts_str2[i] = pts_str[i + p - pts_str + 1];
					}
					pts_str2[i] = 0;
					husen_points.push_back(Point(atoi(pts_str1), atoi(pts_str2)));
					if (husen_points.size() == 4) {
						similar_husens.push_back(husen_points);
						husen_center_data.push_back(fmean2i(&husen_points[0], 4, 0));
						husen_points_data.push_back(husen_points);
						husen_points = vector<Point>();
					}
				}
			}
		}
		all_category_data[i] = team_category_data;
		(*all_A_).push_back(team_A);
		(*all_husen_center_data_).push_back(team_husen_center_data);
		(*all_husen_points_data_).push_back(team_husen_points_data);
		/*
		all_A[i] = team_A;
		all_husen_center_data[i] = team_husen_center_data;
		all_husen_points_data[i] = team_husen_points_data;
		*/
	}
	//* min_dict_keisan
	FILE *fp;
	fopen_s(&fp, "husen_dict.csv", "w");
	int group;
	int sample;
	for (group = 0; group < num_group; group++) {
		for (sample = 0; sample < num_sample; sample++) {
			vector<vector<vector<Point>>>category_data = all_category_data[group][sample];
			int i, j, k, l;
			for (i = 0; i < category_data.size(); i++) {
				for (j = 0; j < category_data[i].size(); j++) {
					Point2f center1 = fmean2i(&category_data[i][j][0], 4, 0);
					double min_dict_same = -1;
					double min_dict_other = -1;
					double min_fusen_dict_same = -1;
					double min_fusen_dict_other = -1;
					double dict;
					for (l = 0; l < category_data[i].size(); l++) {
						if (l != j) {
							Point2f center2 = fmean2i(&category_data[i][l][0], 4, 0);
							dict = norm(center2 - center1);
							if (min_dict_same < 0 || dict < min_dict_same) {
								min_dict_same = dict;
							}
							//fusen dict
							dict = fusen_dict(category_data[i][j], category_data[i][l]);
							if (min_fusen_dict_same < 0 || dict < min_fusen_dict_same) {
								min_fusen_dict_same = dict;
							}
						}
					}
					for (k = 0; k < category_data.size(); k++) {
						if (k != i) {
							for (l = 0; l < category_data[k].size(); l++) {
								Point2f center2 = fmean2i(&category_data[k][l][0], 4, 0);
								dict = norm(center2 - center1);
								if (min_dict_other < 0 || dict < min_dict_other) {
									min_dict_other = dict;
								}
								//fusen dict
								double dict = fusen_dict(category_data[i][j], category_data[k][l]);
								if (min_fusen_dict_other < 0 || dict < min_fusen_dict_other) {
									min_fusen_dict_other = dict;
								}

							}
						}
					}
					//cout << min_dict_same << endl;
					//cout << min_dict_other << endl;
					//cout << min_fusen_dict_same << endl;
					//cout << min_fusen_dict_other << endl;
					fprintf(fp, "%d, %d, %lf,%lf,%lf,%lf\n", group + 1, sample + 1, min_dict_same, min_dict_other, min_fusen_dict_same, min_fusen_dict_other);
				}
			}
		}
	}
	fclose(fp);
	//*/

	/*
	*all_A_ = all_A;
	*all_husen_center_data_ = all_husen_center_data;
	*all_husen_points_data_ = all_husen_points_data;
	*/

	return 0;
	/*
	vector<vector<vector<vector<Point>>>> mus(4);
	vector<vector<Point>> me;
	vector<Point>points;
	FILE*fp[4];
	fopen_s(&fp[0], "mi1.csv", "r");
	fopen_s(&fp[1], "mi2.csv", "r");
	fopen_s(&fp[2], "mi3.csv", "r");
	fopen_s(&fp[3], "mi4.csv", "r");
	int fnum;
	for (fnum = 0; fnum < 4; fnum++) {

	}



	vector<vector<vector<Point>>>datas(4);
	vector<vector<Point2f>>datas_center(4);
	vector<vector<vector<int>>> As(4);
	int i, j;
	for (fnum = 0; fnum < 4; fnum++) {
	int count = 0;
	for (i = 0; i < mus[fnum].size(); i++) {
	As[fnum].push_back(vector<int>());
	for (j = 0; j < mus[fnum][i].size(); j++) {
	As[fnum][i].push_back(count);
	count++;
	datas[fnum].push_back(mus[fnum][i][j]);
	datas_center[fnum].push_back(fmean2i(&mus[fnum][i][j][0], 4, 0));
	}
	}
	}
	return 0;
	//*/
}


double calcF(vector<vector<int>> A, vector<vector<int>>C, int N) {
	vector<vector<int>>X = calc_cross(A, C);
	//double entropy = calc_entropy(X, N);
	//double purity = calc_purity(X, N);
	double F = calc_F(X, N);
	return F;

}
double calcFp(vector<vector<int>> A, vector<vector<int>>C, int N) {
	vector<vector<int>>X = calc_cross(A, C);
	//double entropy = calc_entropy(X, N);
	//double purity = calc_purity(X, N);
	//double F = calc_Fp(X, N);
	double p = calcP(A, C, N);
	double r = calcR(A, C, N);
	return 2 * p*r / (p + r);
}
double calcR(vector<vector<int>> A, vector<vector<int>>C, int N) {
	vector<vector<int>>X = calc_cross(A, C);
	//double entropy = calc_entropy(X, N);
	//double purity = calc_purity(X, N);
	double R = calc_R(X, N);
	return R;

}
int mitukeru(int id, vector<vector<int>> C)
{
	int i, j;
	for (i = 0; i < C.size(); i++) {
		for (j = 0; j < C[i].size(); j++) {
			if (id == C[i][j]) {
				return i;
			}
		}
	}
	return C.size() + 1;
}
double calcP(vector<vector<int>> A, vector<vector<int>>C, int N) {
	vector<vector<int>>X = calc_cross(A, C);
	//double entropy = calc_entropy(X, N);
	//double purity = calc_purity(X, N);
	double P = calc_P(X, N);
	return P;

}

template <class X> X modo(X* start, int size) {

	int max = 0, cnt = 0, index;

	for (int j = 0; j<size; j++, cnt = 0)
	{
		for (int i = j; i<size; i++)
		{
			if (start[j] == start[i])
				cnt++;
		}
		if (max < cnt) {
			max = cnt;
			index = j;
		}
	}
	return start[index];
}

