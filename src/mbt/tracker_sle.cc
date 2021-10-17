#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <opencv2/highgui.hpp>
#include <glog/logging.h>

#include "base/global_param.hh"
#include "mbt/m_func.hh"
#include "mbt/histogram.hh"
#include "mbt/search_line.hh"
#include "mbt/tracker_sle.hh"

enum {
	RUN_TRACK = 0,
	RUN_DEBUG = 1,
};

SLETracker::SLETracker(const cv::Matx33f& K, std::vector<Object3D*>& objects)
	: SLTracker(K, objects)
{

}

void SLETracker::ComputeJac(
	Object3D* object,
	int m_id, 
	const cv::Mat& frame,
	const cv::Mat& depth_map,
	const cv::Mat& depth_inv_map, 
	cv::Matx66f& wJTJM, cv::Matx61f& JTM) 
{
	float* depth_data = (float*)depth_map.ptr<float>();
	float* depth_inv_data = (float*)depth_inv_map.ptr<float>();
	uchar* frame_data = frame.data;
	const std::vector<std::vector<cv::Point> >& search_points = search_line->search_points;
	const std::vector<std::vector<cv::Point2f> >& bundle_prob = search_line->bundle_prob;

	JTM = cv::Matx61f::zeros();
	wJTJM = cv::Matx66f::zeros();
	float* JT = JTM.val;
	float* wJTJ = wJTJM.val;

	float zf = view->getZFar();
	float zn = view->getZNear();
	cv::Matx33f K = view->GetCalibrationMatrix().get_minor<3, 3>(0, 0);
	float* K_inv_data =  K.inv().val;
	float fx = K(0,0);
	float fy = K(1,1);

	for (int r = 0; r < search_points.size(); r++) {
		if (!search_line->actives[r])
			continue;

		int eid = search_points[r][search_points[r].size()-1].y;
		if (eid < 0)
			continue;

		int last = search_points[r].size()-2;
		float nx = search_points[r][last].x - search_points[r][0].x;
		float ny = search_points[r][last].y - search_points[r][0].y;
		float dnxy = 1.0f / sqrt(nx*nx + ny*ny);
		nx *= -dnxy;
		ny *= -dnxy;

		int mid = search_points[r][search_points[r].size() - 1].x;
		int mx = search_points[r][mid].x;
		int my = search_points[r][mid].y;
		int zidx = my * depth_map.cols + mx;
		float ex = nx * (search_points[r][mid].x - search_points[r][eid].x) + ny * (search_points[r][mid].y - search_points[r][eid].y);
		
		float we = tukey_weight(ex, 10) * ecweight(scores[r], 1.0f);

		float depth = 1.0f - depth_data[zidx];
		float D = 2.0f * zn * zf / (zf + zn - (2.0f * depth - 1.0) * (zf - zn));
		float Xc = D * (K_inv_data[0] * mx + K_inv_data[2]);
		float Yc = D * (K_inv_data[4] * my + K_inv_data[5]);
		float Zc = D;
		float Zc2 = Zc*Zc;

		float J[6];

		J[0] = nx * (-Xc*fx*Yc/Zc2) +     ny * (-fy -Yc*Yc*fy/Zc2);
		J[1] = nx * (fx + Xc*Xc*fx/Zc2) + ny * (Xc*Yc*fy/Zc2);
		J[2] = nx * (-fx*Yc/Zc)+          ny * (Xc*fy/Zc);
		J[3] = nx * (fx/Zc);
		J[4] =                            ny * (fy/Zc);
		J[5] = nx * (-Xc*fx/Zc2) +        ny * (-Yc*fy/Zc2);

		for (int n = 0; n < 6; n++) {
			JT[n] += we * ex * J[n];
		}

		for (int n = 0; n < 6; n++)
		for (int m = n; m < 6; m++) {
			wJTJ[n * 6 + m] += we * J[n] * J[m];
		}

		depth = 1.0f - depth_inv_data[zidx];
		D = 2.0f * zn * zf / (zf + zn - (2.0f * depth - 1.0) * (zf - zn));
		Xc = D * (K_inv_data[0] * mx + K_inv_data[2]);
		Yc = D * (K_inv_data[4] * my + K_inv_data[5]);
		Zc = D;
		Zc2 = Zc * Zc;

		J[0] = nx * (-Xc*fx*Yc/Zc2) +     ny * (-fy -Yc*Yc*fy/Zc2);
		J[1] = nx * (fx + Xc*Xc*fx/Zc2) + ny * (Xc*Yc*fy/Zc2);
		J[2] = nx * (-fx*Yc/Zc)+          ny * (Xc*fy/Zc);
		J[3] = nx * (fx/Zc);
		J[4] =                            ny * (fy/Zc);
		J[5] = nx * (-Xc*fx/Zc2) +        ny * (-Yc*fy/Zc2);

		for (int n = 0; n < 6; n++) {
			JT[n] += we * ex * J[n];
		}

		for (int n = 0; n < 6; n++)
		for (int m = n; m < 6; m++) {
			wJTJ[n * 6 + m] += we * J[n] * J[m];
		}
	}

	for (int i = 0; i < wJTJM.rows; i++)
	for (int j = i + 1; j < wJTJM.cols; j++) {
		wJTJM(j, i) = wJTJM(i, j);
	}
}

void SLETracker::FindMatchPointMaxProb(float diff) {
	std::vector<std::vector<cv::Point> >& search_points = search_line->search_points;
	const std::vector<std::vector<cv::Point2f> >& bundle_prob = search_line->bundle_prob;
	scores.resize(search_points.size());


	for (int r = 0; r < bundle_prob.size(); ++r) {
		float prob_max = 0.0f;
		search_points[r][search_points[r].size()-1].y = -1;
		scores[r] = 0.0f;

		int mid = search_points[r][search_points[r].size() - 1].x;

		float nx = search_line->norms[r].x;
		float ny = search_line->norms[r].y;

		for (int c = 3; c < bundle_prob[r].size()-3; ++c) {
			if (bundle_prob[r][c + 1].x - bundle_prob[r][c - 1].x > diff) {
				float prbf = 
					bundle_prob[r][c-3].x*
					bundle_prob[r][c-2].x*
					bundle_prob[r][c-1].x;
				float prbb = 
					bundle_prob[r][c-3].y*
					bundle_prob[r][c-2].y*
					bundle_prob[r][c-1].y;
				float prff = 
					bundle_prob[r][c+1].x*
					bundle_prob[r][c+2].x*
					bundle_prob[r][c+3].x;
				float prfb = 
					bundle_prob[r][c+1].y*
					bundle_prob[r][c+2].y*
					bundle_prob[r][c+3].y;

				float pr_C = prbb*prff;
				float pr_F = prbf*prff;
				float pr_B = prbb*prfb;

				// both tukey_weight(ex, 10) * slweight(pr_C, 1.0f); cam_regular 89.3
				if ((pr_C > pr_F) && (pr_C > pr_B)) {

					float ex = nx * (search_points[r][mid].x - search_points[r][c].x) + ny * (search_points[r][mid].y - search_points[r][c].y);
					float we = tukey_weight(ex, 10) * ecweight(pr_C, 1.0f);

					if (we > prob_max) {
						prob_max = we;
						scores[r] = pr_C;
						search_points[r][search_points[r].size()-1].y = c;
					}
				}
			}
		}
	}
}

void SLETracker::Track(std::vector<cv::Mat>& imagePyramid, std::vector<Object3D*>& objects, int runs) {
	for (int iter = 0; iter < runs * 4; iter++) {
		RunIteration(objects, imagePyramid, 2, 12, 2);
	}

	for (int iter = 0; iter < runs * 2; iter++) {
		RunIteration(objects, imagePyramid, 1, 12, 2);
	}

	for (int iter = 0; iter < runs * 1; iter++) {
		RunIteration(objects, imagePyramid, 0, 12, 2);
	}
}

void SLETracker::RunIteration(std::vector<Object3D*>& objects, const std::vector<cv::Mat>& imagePyramid, int level, int sl_len, int sl_seg, int run_type) {
	int width = view->GetWidth();
	int height = view->GetHeight();
	view->setLevel(level);
	int numInitialized = 0;
	for (int o = 0; o < objects.size(); o++) {
		if (!objects[o]->isInitialized())
			continue;

		numInitialized++;

		cv::Rect roi = Compute2DROI(objects[o], cv::Size(width / pow(2, level), height / pow(2, level)), 8);
		if (roi.area() == 0)
			continue;

		while (roi.area() < 3000 && level > 0) {
			level--;
			view->setLevel(level);
			roi = Compute2DROI(objects[o], cv::Size(width / pow(2, level), height / pow(2, level)), 8);
		}
	}

	view->setLevel(level);
	view->RenderSilhouette(std::vector<Model*>(objects.begin(), objects.end()), GL_FILL);
	cv::Mat depth_map = view->DownloadFrame(View::DEPTH);

	cv::Mat masks_map;
	if (numInitialized > 1) {
		masks_map = view->DownloadFrame(View::MASK);
	}	else {
		masks_map = depth_map;
	}

	for (int o = 0; o < objects.size(); o++) {
		if (!objects[o]->isInitialized())  
			continue;

		cv::Rect roi = Compute2DROI(objects[o], cv::Size(width / pow(2, level), height / pow(2, level)), 8);
		if (roi.area() == 0)
			continue;

		int m_id = (numInitialized <= 1) ? -1 : objects[o]->getModelID();
		cv::Mat mask_map;
		ConvertMask(masks_map, m_id, roi, mask_map);

		search_line->FindSearchLine(mask_map, imagePyramid[level], sl_len, sl_seg, true);

		if (numInitialized > 1) {
			FilterOccludedPoint(masks_map, depth_map);
		}

		GetBundleProb(imagePyramid[level], o);

		FindMatchPointMaxProb(0.2);

		view->RenderSilhouette(objects[o], GL_FILL, true);
		cv::Mat depth_inv_map = view->DownloadFrame(View::DEPTH);

		cv::Matx66f wJTJ;
		cv::Matx61f JT;
		ComputeJac(objects[o], m_id, imagePyramid[level], depth_map, depth_inv_map, wJTJ, JT);

		cv::Matx44f T_cm = Transformations::exp(-wJTJ.inv(cv::DECOMP_CHOLESKY)*JT)* objects[o]->getPose();
		objects[o]->setPose(T_cm);
	}
}
