#include "camera.hh"

Camera* Camera::BuildCamera(const std::string& frames) {
	//struct _stat buf;
	struct stat buf;

	Camera* grabber = NULL;
	//if (0 != _stat(frames.c_str(), &buf)) {
	if (0 != stat(frames.c_str(), &buf)) {
		grabber = new VideoLoaderCamera(0);
	//} else if(_S_IFDIR & buf.st_mode) {
	} else if(S_ISDIR(buf.st_mode)) {
		grabber = new ImageLoaderCamera(frames);
	} else {
		grabber = new VideoLoaderCamera(frames);
	}

	if (-1 == grabber->width) {
		delete grabber;
		return NULL;
	}	else {
		return grabber;
	}
}

const cv::Mat& Camera::image() const { return image_; }