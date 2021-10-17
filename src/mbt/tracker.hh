#pragma once

#include <opencv2/core.hpp>

#include "mbt/object3d.hh"
#include "mbt/signed_distance_transform2d.hh"
#include "mbt/template_view.hh"
#include "mbt/camera.hh"

class Viewer;

class Tracker {
public:
	//Tracker(const cv::Matx33f& K, const cv::Matx14f& distCoeffs, std::vector<Object3D*>& objects);
	Tracker(const cv::Matx33f& K, std::vector<Object3D*>& objects);
	void Init(std::shared_ptr<Camera> camera_ptr) {
		camera_ptr_ = std::move(camera_ptr);
	}

	static Tracker* GetTracker(int id, const cv::Matx33f& K, const cv::Matx14f& distCoeffs, std::vector<Object3D*>& objects);

	virtual void ToggleTracking(int objectIndex, bool undistortFrame = true);
	virtual void EstimatePoses(bool check_lost);
	virtual void PreProcess() {}
	virtual void PostProcess() {}

	void reset();

	int64 render_time;
	int64 sl_time;
	int64 jac_time;
	int64 sc_time;
	int64 pp_time;

	void AddViewer(std::shared_ptr<Viewer> viewer_ptr);
	bool UpdateViewers(int save_idx);

protected:
	virtual void Track(std::vector<cv::Mat>& imagePyramid, std::vector<Object3D*>& objects, int runs = 1) = 0;

	void CheckPose(std::vector<Object3D*>& objects);

	cv::Rect Compute2DROI(Object3D* object, const cv::Size& maxSize, int offset);
	cv::Rect computeBoundingBox(const std::vector<cv::Point3i>& centersIDs, int offset, int level, const cv::Size& maxSize);



	static void ConvertMask(const cv::Mat& maskm, uchar oid, cv::Mat& mask);
	static void ConvertMask(const cv::Mat& maskm, uchar oid, cv::Rect& roi, cv::Mat& mask);
	static void ShowMask(const cv::Mat& masks, cv::Mat& buf);

protected:
	std::vector<std::shared_ptr<Viewer>> viewer_ptrs_;

	std::vector<Object3D*> objects;
	
	std::shared_ptr<Camera> camera_ptr_ = nullptr;

	View* view;
	cv::Matx33f K;
	//cv::Matx14f distCoeffs;
	//cv::Mat map1;
	//cv::Mat map2;

	bool initialized;
};

class Histogram;

class TrackerBase : public Tracker {
public:
	TrackerBase(const cv::Matx33f& K, std::vector<Object3D*>& objects);

	virtual void PreProcess() override;
	virtual void PostProcess() override;
	virtual void UpdateHist();

protected:
	void DetectEdge(const cv::Mat& img, cv::Mat& edge_map);
	
protected:
	Histogram* hists;
};

class SearchLine;

class SLTracker: public TrackerBase {
public:
	SLTracker(const cv::Matx33f& K, std::vector<Object3D*>& objects);

	void GetBundleProb(const cv::Mat& frame, int oid);
	void FilterOccludedPoint(const cv::Mat& mask, const cv::Mat& depth);

protected:
	std::shared_ptr<SearchLine> search_line;
	std::vector<float> scores;
};

inline float GetDistance(const cv::Point& p1, const cv::Point& p2) {
	float dx = float(p1.x - p2.x);
	float dy = float(p1.y - p2.y);
	return sqrt(dx*dx + dy*dy);
}

/**
 *  This class extends the OpenCV ParallelLoopBody for efficiently parallelized
 *  computations. Within the corresponding for loop, the RGB values per pixel
 *  of a color input image are converted to their corresponding histogram bin
 *  index.
 */
class Parallel_For_convertToBins : public cv::ParallelLoopBody
{
private:
	cv::Mat _frame;
	cv::Mat _binned;

	uchar* frameData;
	int* binnedData;

	int _numBins;

	int _binShift;

	int _threads;

public:
	Parallel_For_convertToBins(const cv::Mat& frame, cv::Mat& binned, int numBins, int threads)
	{
		_frame = frame;

		binned.create(_frame.rows, _frame.cols, CV_32SC1);
		_binned = binned;

		frameData = _frame.data;
		binnedData = (int*)_binned.ptr<int>();

		_numBins = numBins;

		_binShift = 8 - log(numBins) / log(2);

		_threads = threads;
	}

	virtual void operator()(const cv::Range& r) const
	{
		int range = _frame.rows / _threads;

		int yEnd = r.end * range;
		if (r.end == _threads)
		{
			yEnd = _frame.rows;
		}

		for (int y = r.start * range; y < yEnd; y++)
		{
			uchar* frameRow = frameData + y * _frame.cols * 3;
			int* binnedRow = binnedData + y * _binned.cols;

			int idx = 0;
			for (int x = 0; x < _frame.cols; x++, idx += 3)
			{
				int ru = (frameRow[idx] >> _binShift);
				int gu = (frameRow[idx + 1] >> _binShift);
				int bu = (frameRow[idx + 2] >> _binShift);

				int binIdx = (ru * _numBins + gu) * _numBins + bu;

				binnedRow[x] = binIdx;
			}
		}
	}
};