#include <opencv2/highgui.hpp>

#include "mbt/viewer.hh"
#include "mbt/model.hh"
#include "mbt/view.hh"

void Viewer::StartSavingImages(const std::filesystem::path& path) {
  save_images_ = true;
  save_path_ = path;
}

void Viewer::StopSavingImages() { 
	save_images_ = false; 
}

void Viewer::set_display_images(bool dispaly_images) {
	display_images_ = dispaly_images;
}

void Viewer::PrintID(int fid, cv::Mat& frame) {
	std::stringstream sstr;
	sstr << "#" << std::setw(3) << std::setfill('0') << fid;
	cv::putText(frame, sstr.str(), cv::Point(5, 20), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, cv::Scalar(255, 255, 255), 1);
}

void Viewer::ShowImage(const cv::Mat& frame) {
	if (frame.cols > 1440) {
		cv::Mat small_out;
		cv::resize(frame, small_out, cv::Size(frame.cols * 0.75, frame.rows * 0.75));
		cv::imshow(name_, small_out);
	}	else {
		cv::imshow(name_, frame);
	}
}

void ImageViewer::Init(const std::string& name) {
  name_ = name;
  initialized_ = true;
}

void ImageViewer::UpdateViewer(int save_index) {
	auto frame = camera_ptr_->image();
  if (display_images_)
		ShowImage(frame);

  if (save_images_)
    cv::imwrite(
      save_path_.string() + name_ + "_" + std::to_string(save_index) + ".png",
      frame);
}

void ContourViewer::Init(const std::string& name, View* view, const std::vector<Model*>& objects, std::shared_ptr<Camera> camera_ptr) {
	camera_ptr_ = std::move(camera_ptr);
	renderer_ = view;
	objects_ = objects;
	name_ = name;
  initialized_ = true;
}

void ContourViewer::UpdateViewer(int fid) {
	auto frame = camera_ptr_->image();
	auto contour_img = DrawContourOverlay(renderer_, objects_, frame);
	PrintID(fid, contour_img);

	if (display_images_)
		ShowImage(contour_img);
  
  if (save_images_)
    cv::imwrite(
      save_path_.string() + name_ + "_" + std::to_string(fid) + ".png",
			contour_img);
}

cv::Mat ContourViewer::DrawContourOverlay(View* view, const std::vector<Model*>& objects, const cv::Mat& frame) {
	view->setLevel(0);
	view->RenderSilhouette(std::vector<Model*>(objects.begin(), objects.end()), GL_FILL);

	cv::Mat depth_map = view->DownloadFrame(View::DEPTH);
	cv::Mat masks_map;
	if (objects.size() > 1) {
		masks_map = view->DownloadFrame(View::MASK);
	}	else {
		masks_map = depth_map;
	}

	cv::Mat result = frame.clone();

	for (int oid = 0; oid < objects.size(); oid++) {
		cv::Mat mask_map;
		view->ConvertMask(masks_map, mask_map, objects[oid]->getModelID());

		std::vector<std::vector<cv::Point> > contours;
		cv::findContours(mask_map, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

		cv::Vec3b color;
		if (0 == oid)
			color = cv::Vec3b(0, 255, 0);
		if (1 == oid)
			color = cv::Vec3b(0, 0, 255);

		for (auto contour : contours)
		for (auto pt : contour) {
			result.at<cv::Vec3b>(pt) = color;
		}
	}

	return result;
}

void MeshViewer::Init(const std::string& name, View* view, const std::vector<Model*>& objects, std::shared_ptr<Camera> camera) {
	renderer_ = view;
	objects_ = objects;
	name_ = name;
	initialized_ = true;
}

void MeshViewer::UpdateViewer(int fid) {
	auto frame = camera_ptr_->image();
	auto res_img = DrawMeshOverlay(renderer_, objects_, frame);
	PrintID(fid, res_img);

	if (display_images_)
		ShowImage(res_img);

	if (save_images_)
		cv::imwrite(
			save_path_.string() + name_ + "_" + std::to_string(fid) + ".png",
			res_img);
}

cv::Mat MeshViewer::DrawMeshOverlay(View* view, const std::vector<Model*>& objects, const cv::Mat& frame) {
	view->setLevel(0);

	std::vector<cv::Point3f> colors;
	colors.push_back(cv::Point3f(1.0, 0.5, 0.0));
	colors.push_back(cv::Point3f(0.2, 0.3, 1.0));
	//RenderShaded(std::vector<Model*>(objects.begin(), objects.end()), GL_FILL, colors, true);
	//RenderNormals(std::vector<Model*>(objects.begin(), objects.end()), GL_FILL);
	cv::Mat result = frame.clone();
	view->RenderCV(objects[0], result);
	//RenderCV(objects[0], result, cv::Scalar(1,255,1));
	//RenderCV(objects[1], result, cv::Scalar(1,1,255));
	return result;
}

void FragmentViewer::Init(const std::string& name, View* view, const std::vector<Model*>& objects, std::shared_ptr<Camera> camera) {
	renderer_ = view;
	objects_ = objects;
	name_ = name;
	initialized_ = true;
}

void FragmentViewer::UpdateViewer(int fid) {
	const cv::Mat& frame = camera_ptr_->image();
	auto res_img = DrawFragmentOverlay(renderer_, objects_, frame);
	PrintID(fid, res_img);

	if (display_images_)
		ShowImage(res_img);

	if (save_images_)
		cv::imwrite(
			save_path_.string() + name_ + "_" + std::to_string(fid) + ".png",
			res_img);
}

cv::Mat FragmentViewer::DrawFragmentOverlay(View* view, const std::vector<Model*>& objects, const cv::Mat& frame) {
	// render the models with phong shading
	view->setLevel(0);

	std::vector<cv::Point3f> colors;
	colors.push_back(cv::Point3f(1.0, 0.5, 0.0));
	//colors.push_back(cv::Point3f(0.0, 1.0, 0.0));
	//colors.push_back(Point3f(0.2, 0.3, 0.0));
	view->RenderShaded(std::vector<Model*>(objects.begin(), objects.end()), GL_FILL, colors, true);
	//RenderNormals(std::vector<Model*>(objects.begin(), objects.end()), GL_FILL);

	// download the rendering to the CPU
	cv::Mat rendering = view->DownloadFrame(View::RGB);

	// download the depth buffer to the CPU
	cv::Mat depth = view->DownloadFrame(View::DEPTH);

	// compose the rendering with the current camera image for demo purposes (can be done more efficiently directly in OpenGL)
	cv::Mat result = frame.clone();
	for (int y = 0; y < frame.rows; y++)
	for (int x = 0; x < frame.cols; x++) {
		cv::Vec3b color = rendering.at<cv::Vec3b>(y, x);
		if (depth.at<float>(y, x) != 0.0f) {
			result.at<cv::Vec3b>(y, x)[0] = color[2];
			result.at<cv::Vec3b>(y, x)[1] = color[1];
			result.at<cv::Vec3b>(y, x)[2] = color[0];
		}
	}
	return result;
}