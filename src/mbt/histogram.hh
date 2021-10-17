#pragma once
#include <vector>
#include "mbt/object3d.hh"
#include "mbt/tclc_histograms.hh"
//#include "mbt/wtclc_histograms.hh"

class View;
class SearchLine;

class Histogram {
public:
	Histogram();
	virtual ~Histogram() = 0;

	virtual void Update(const cv::Mat& frame, cv::Mat& mask_map, cv::Mat& depth_map, int oid, float afg, float abg)  = 0;
	virtual void GetPixelProb(uchar rc, uchar gc, uchar bc, int x, int y, int oid, float& ppf, float& ppb) = 0;
	virtual void GetRegionProb(const cv::Mat& frame, int oid, cv::Mat& prob_map) = 0;

protected:
	View* view;
	
};

class RBOTHist : public Histogram {
public:
	RBOTHist(const std::vector<Object3D*>& objects);

	virtual void Update(const cv::Mat& frame, cv::Mat& mask_map, cv::Mat& depth_map, int oid, float afg, float abg) override;
	virtual void GetPixelProb(uchar rc, uchar gc, uchar bc, int x, int y, int oid, float& ppf, float& ppb) override;
	virtual void GetRegionProb(const cv::Mat& frame, int oid, cv::Mat& prob_map) override;

protected:
	std::vector<Object3D*> objs;
};