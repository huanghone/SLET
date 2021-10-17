#include <cmath>
#include <iostream>
#include <sstream>
#include <fstream>
#include <QApplication>
#include <QThread>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#include "base/pose_io.hh"
#include "base/global_param.hh"
#include "base/util.hh"
#include "mbt/camera.hh"
#include "mbt/object3d.hh"
#include "mbt/view.hh"
#include "mbt/tracker.hh"
#include "mbt/viewer.hh"

int Track(const std::string& config_file) {
	std::cout << std::endl << "#######################" << std::endl;
	tk::GlobalParam* gp = tk::GlobalParam::Instance();
	gp->ParseConfig(config_file);

	std::shared_ptr<Camera> camera_ptr(Camera::BuildCamera(gp->frames));
	CHECK(camera_ptr) << "ERROR: check <frames> in config file.";
	gp->image_width = camera_ptr->width;
	gp->image_height = camera_ptr->height;
	cv::Matx33f K = cv::Matx33f(gp->fx, 0, gp->cx, 0, gp->fy, gp->cy, 0, 0, 1);
	cv::Matx14f distCoeffs = cv::Matx14f(0.0, 0.0, 0.0, 0.0);

	std::unique_ptr<PoseReader> pose_reader(new PoseReaderRBOT);

	std::vector<std::vector<cv::Matx44f> > gt_poses;
	pose_reader->Read(gp->gt_pose_file, gt_poses);

	int init_fid = gp->target_frame - 1 >= 0 ? gp->target_frame - 1 : gp->target_frame;
	std::vector<float> distances = { 200.0f, 400.0f, 600.0f };
	std::vector<Object3D*> objects;
	for (int i = 0; i < gp->model_file.size(); ++i) {
		objects.push_back(new Object3D(gp->model_file[i], gt_poses[i][init_fid], 1.0, 0.55f, distances));
		objects[i]->fcount = 0;
	}

	View* view = View::Instance();
	view->init(K, gp->image_width, gp->image_height, gp->zn, gp->zf, 4);

	//PoseWriter pose_writer(config_file.substr(0, config_file.size()-4)+".tk", config_file, objects.size());
	PoseWriter pose_writer(gp->tk_pose_file, config_file, objects.size());
	ReportWriter report_writer(config_file, gp->report_file);
	LostWriter lost_wirter(config_file.substr(0, config_file.size()-4) + ".lost");
	//FpsWriter fps_writer(gp->fps_file);

	std::shared_ptr<Tracker> tracker_ptr(Tracker::GetTracker(gp->tracker_mode, K, distCoeffs, objects));
	tracker_ptr->Init(camera_ptr);
#if 0
	auto contour_viewer_ptr = std::make_shared<ContourViewer>();
	contour_viewer_ptr->Init("contour viewer", view, std::vector<Model*>(objects.begin(), objects.end()), camera_ptr);
	tracker_ptr->AddViewer(contour_viewer_ptr);
	//contour_viewer_ptr->set_display_images(false);

	auto viewer_ptr_ = std::make_shared<ImageViewer>();
	viewer_ptr_->Init("normal viewer");
	tracker_ptr->AddViewer(viewer_ptr_);

	auto mesh_viewer_ptr_ = std::make_shared<MeshViewer>();
	mesh_viewer_ptr_->Init("mesh viewer", view, std::vector<Model*>(objects.begin(), objects.end()));
	tracker_ptr->AddViewer(mesh_viewer_ptr_);

	auto fragment_viewer_ptr_ = std::make_shared<FragmentViewer>();
	fragment_viewer_ptr_->Init("fragment viewer", view, std::vector<Model*>(objects.begin(), objects.end()));
	tracker_ptr->AddViewer(fragment_viewer_ptr_);
#endif
	int timeout = gp->timeout;
	
	if (camera_ptr->UpdateCamera()) {
		for (int oid = 0; oid < objects.size(); ++oid)
			tracker_ptr->ToggleTracking(oid, false);
		tracker_ptr->PreProcess();
	}	else {
		LOG(ERROR) << "can not read frame";
		return -1;
	}

	int64 hist_time = 0;
	int64 all_time = 0;

	int fid = gp->target_frame;
	CHECK(fid >= 0);
	for (;;fid++) {
		if (!camera_ptr->UpdateCamera())
			break;

		tracker_ptr->EstimatePoses(gp->bench_mode);
		tracker_ptr->UpdateViewers(fid);

		for (int i = 0; i < objects.size(); ++i) {
			cv::Matx44f tk_pose = objects[i]->getPose();
			pose_writer.Record(tk_pose, fid);

			if (gp->bench_mode && pose_reader->IsLostGT(gt_poses[i][fid], tk_pose, objects[i])) {
				objects[i]->setPose(gt_poses[i][fid]);
				objects[i]->fcount++;
				if (0 == i)
					lost_wirter.Record(fid);
				std::cout << "LOST [" << i << "]: "<< fid << std::endl;
			}
		}

		tracker_ptr->PostProcess();

		int key = cv::waitKey(timeout);
		if (27 == key)
			break;
		if ('r' == key) {
			for (int i = 0; i < objects.size(); ++i) {
				objects[i]->setPose(gt_poses[i][fid]);
			}
		}
	}

	int frame_count = fid - gp->target_frame;

	float avg_rt = all_time * 1000.0f / cv::getTickFrequency()  / frame_count;
	if (gp->bench_mode) {
		float score = 1 - (float)objects[0]->fcount / (float)frame_count;
		report_writer.Record(score*100.f, avg_rt);

		std::cout.setf(std::ios_base::fixed, std::ios_base::floatfield);
		std::cout.setf(std::ios_base::showpoint);
		std::cout.precision(1);
		std::cout << "SCORE: " << score*100.f << std::endl;
	}

	std::cout << "#######################" << std::endl;

	View::Instance()->destroy();

	for (int i = 0; i < objects.size(); i++) {
		delete objects[i];
	}
	objects.clear();
}

int main(int argc, char* argv[]) {
	QApplication a(argc, argv);
	Track(argv[1]);
	return 0;
}