#pragma once

#include <filesystem>
#include <string>
#include <opencv2/core.hpp>

#include "mbt/camera.hh"

class Model;

class Viewer {
public:
	virtual void UpdateViewer(int save_index) = 0;
	void StartSavingImages(const std::filesystem::path& path);
	void StopSavingImages();

	void set_display_images(bool dispaly_images);
protected:
	void PrintID(int fid, cv::Mat& frame);
	void ShowImage(const cv::Mat& frame);

	std::shared_ptr<Camera> camera_ptr_ = nullptr;
	std::string name_{};
	std::filesystem::path save_path_{};
	bool display_images_ = true;
	bool save_images_ = false;
	bool initialized_ = false;
};

class ImageViewer : public Viewer {
public:
	void Init(const std::string& name);
	ImageViewer() = default;

	void UpdateViewer(int save_index) override;
};

class View;

class ContourViewer : public Viewer {
public:
	void Init(const std::string& name, View* view, const std::vector<Model*>& objects, std::shared_ptr<Camera> camera);
	ContourViewer() = default;

	void UpdateViewer(int save_index) override;
	cv::Mat DrawContourOverlay(View* view, const std::vector<Model*>& objects, const cv::Mat& frame);

protected:
	View* renderer_;
	std::vector<Model*> objects_;
};

class MeshViewer : public Viewer {
public:
	void Init(const std::string& name, View* view, const std::vector<Model*>& objects, std::shared_ptr<Camera> camera);
	MeshViewer() = default;

	void UpdateViewer(int save_index) override;
	cv::Mat DrawMeshOverlay(View* view, const std::vector<Model*>& objects, const cv::Mat& frame);

protected:
	View* renderer_;
	std::vector<Model*> objects_;
};

class FragmentViewer : public Viewer {
public:
	void Init(const std::string& name, View* view, const std::vector<Model*>& objects, std::shared_ptr<Camera> camera);
	FragmentViewer() = default;

	void UpdateViewer(int save_index) override;
	cv::Mat DrawFragmentOverlay(View* view, const std::vector<Model*>& objects, const cv::Mat& frame);

protected:
	View* renderer_;
	std::vector<Model*> objects_;
};