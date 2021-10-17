#include <iostream>
#include <opencv2/highgui.hpp>

#include "tclc_histograms.hh"
#include "model.hh"

using namespace std;
using namespace cv;

TCLCHistograms::TCLCHistograms(Model *model, int numBins, int radius, float offset)
{
    this->_model = model;
    
    this->numBins = numBins;
    
    this->radius = radius;
    
    this->_offset = offset;
    
    this->_numHistograms = _model->getNumSimpleVertices();
    
    normalizedFG = Mat::zeros(this->_numHistograms, numBins*numBins*numBins, CV_32FC1);
    normalizedBG = Mat::zeros(this->_numHistograms, numBins*numBins*numBins, CV_32FC1);
    
    notNormalizedFG = Mat::zeros(300, numBins*numBins*numBins, CV_32FC1);
    notNormalizedBG = Mat::zeros(300, numBins*numBins*numBins, CV_32FC1);
    
    sumsFB = Mat::zeros(this->_numHistograms, 1, CV_32FC2);

    initialized = Mat::zeros(1, this->_numHistograms, CV_8UC1);
}

TCLCHistograms::~TCLCHistograms()
{
    
}

/**
 *  This class extends the OpenCV ParallelLoopBody for efficiently parallelized
 *  computations. Within the corresponding for loop, for every projected histogram center on or
 *  close to the object's contour, a new foreground and background color histogram are computed
 *  within a local circular image region is computed using the Bresenham algorithm to scan the
 *  corresponding pixels.
 */
class Parallel_For_buildLocalHistograms : public cv::ParallelLoopBody
{
private:
  cv::Mat _frame;
  cv::Mat _mask;

  uchar* frameData;
  uchar* maskData;

  size_t frameStep;
  size_t maskStep;

  cv::Size size;

  std::vector<cv::Point3i> _centers;

  int _radius;

  int _numBins;

  int _binShift;

  int histogramSize;

  cv::Mat _sumsFB;

  int _m_id;

  float* localFGData;
  float* localBGData;

  float* _sumsFBData;

  int _threads;

public:
  Parallel_For_buildLocalHistograms(const cv::Mat& frame, const cv::Mat& mask, const std::vector<cv::Point3i>& centers, float radius, int numBins, cv::Mat& localHistogramsFG, cv::Mat& localHistogramsBG, cv::Mat& sumsFB, int m_id, int threads)
  {
    _frame = frame;
    _mask = mask;

    frameData = _frame.data;
    maskData = _mask.data;

    frameStep = _frame.step;
    maskStep = _mask.step;

    size = frame.size();

    _centers = centers;

    _radius = radius;

    _numBins = numBins;

    _binShift = 8 - log(numBins) / log(2);

    histogramSize = localHistogramsFG.cols;

    localFGData = (float*)localHistogramsFG.ptr<float>();
    localBGData = (float*)localHistogramsBG.ptr<float>();

    _sumsFB = sumsFB;

    _m_id = m_id;

    _sumsFBData = (float*)_sumsFB.ptr<float>();

    _threads = threads;
  }

  void processLine(uchar* frameRow, uchar* maskRow, int xl, int xr, float* localHistogramFG, float* localHistogramBG, float* sumFB) const
  {
    uchar* frame_ptr = (uchar*)(frameRow)+3 * xl;

    uchar* mask_ptr = (uchar*)(maskRow)+xl;
    uchar* mask_max_ptr = (uchar*)(maskRow)+xr;

    for (; mask_ptr <= mask_max_ptr; mask_ptr += 1, frame_ptr += 3)
    {
      int pidx;
      int ru, gu, bu;

      ru = (frame_ptr[0] >> _binShift);
      gu = (frame_ptr[1] >> _binShift);
      bu = (frame_ptr[2] >> _binShift);
      pidx = (ru * _numBins + gu) * _numBins + bu;

      if (*mask_ptr == _m_id)
      {
        localHistogramFG[pidx] += 1;
        sumFB[0]++;
      }
      else
      {
        localHistogramBG[pidx] += 1;
        sumFB[1]++;
      }
    }
  }


  virtual void operator()(const cv::Range& r) const
  {
    int range = (int)_centers.size() / _threads;

    int cEnd = r.end * range;
    if (r.end == _threads)
    {
      cEnd = (int)_centers.size();
    }

    for (int c = r.start * range; c < cEnd; c++)
    {
      int err = 0;
      int dx = _radius;
      int dy = 0;
      int plus = 1;
      int minus = (_radius << 1) - 1;

      int olddx = dx;

      cv::Point3i center = _centers[c];

      int inside = center.x >= _radius && center.x < size.width - _radius && center.y >= _radius && center.y < size.height - _radius;

      float* localHistogramFG = localFGData + c * histogramSize;
      float* localHistogramBG = localBGData + c * histogramSize;

      int cID = _centers[c].z;
      float* sumFB = _sumsFBData + cID * 2;
      sumFB[0] = 0;
      sumFB[1] = 0;

      while (dx >= dy)
      {
        int mask;
        int y11 = center.y - dy, y12 = center.y + dy, y21 = center.y - dx, y22 = center.y + dx;
        int x11 = center.x - dx, x12 = center.x + dx, x21 = center.x - dy, x22 = center.x + dy;

        if (inside)
        {
          uchar* frameRow0 = frameData + y11 * frameStep;
          uchar* frameRow1 = frameData + y12 * frameStep;

          uchar* maskRow0 = maskData + y11 * maskStep;
          uchar* maskRow1 = maskData + y12 * maskStep;

          processLine(frameRow0, maskRow0, x11, x12, localHistogramFG, localHistogramBG, sumFB);
          if (y11 != y12) processLine(frameRow1, maskRow1, x11, x12, localHistogramFG, localHistogramBG, sumFB);

          frameRow0 = frameData + y21 * frameStep;
          frameRow1 = frameData + y22 * frameStep;

          maskRow0 = maskData + y21 * maskStep;
          maskRow1 = maskData + y22 * maskStep;

          if (olddx != dx)
          {
            if (y11 != y21) processLine(frameRow0, maskRow0, x21, x22, localHistogramFG, localHistogramBG, sumFB);
            if (y12 != y22) processLine(frameRow1, maskRow1, x21, x22, localHistogramFG, localHistogramBG, sumFB);
          }
        }
        else if (x11 < size.width && x12 >= 0 && y21 < size.height && y22 >= 0)
        {
          x11 = std::max(x11, 0);
          x12 = MIN(x12, size.width - 1);

          if ((unsigned)y11 < (unsigned)size.height)
          {
            uchar* frameRow = frameData + y11 * frameStep;
            uchar* maskRow = maskData + y11 * maskStep;

            processLine(frameRow, maskRow, x11, x12, localHistogramFG, localHistogramBG, sumFB);
          }

          if ((unsigned)y12 < (unsigned)size.height && (y11 != y12))
          {
            uchar* frameRow = frameData + y12 * frameStep;
            uchar* maskRow = maskData + y12 * maskStep;

            processLine(frameRow, maskRow, x11, x12, localHistogramFG, localHistogramBG, sumFB);
          }

          if (x21 < size.width && x22 >= 0 && (olddx != dx))
          {
            x21 = std::max(x21, 0);
            x22 = MIN(x22, size.width - 1);

            if ((unsigned)y21 < (unsigned)size.height)
            {
              uchar* frameRow = frameData + y21 * frameStep;
              uchar* maskRow = maskData + y21 * maskStep;

              processLine(frameRow, maskRow, x21, x22, localHistogramFG, localHistogramBG, sumFB);
            }

            if ((unsigned)y22 < (unsigned)size.height)
            {
              uchar* frameRow = frameData + y22 * frameStep;
              uchar* maskRow = maskData + y22 * maskStep;

              processLine(frameRow, maskRow, x21, x22, localHistogramFG, localHistogramBG, sumFB);
            }
          }
        }

        olddx = dx;

        dy++;
        err += plus;
        plus += 2;

        mask = (err <= 0) - 1;

        err -= minus & mask;
        dx += mask;
        minus -= mask & 2;
      }
    }
  }
};

/**
 *  This class extends the OpenCV ParallelLoopBody for efficiently parallelized
 *  computations. Within the corresponding for loop, each previously computed local foreground
 *  and background color histogram is merged with their normalized temporally consistent
 *  representation based on respective learning rates.
 */
class Parallel_For_mergeLocalHistograms : public cv::ParallelLoopBody
{
private:
  int histogramSize;

  cv::Mat _sumsFB;

  float* notNormalizedFGData;
  float* notNormalizedBGData;

  float* normalizedFGData;
  float* normalizedBGData;

  uchar* initializedData;

  std::vector<cv::Point3i> _centersIds;

  float _alphaF;
  float _alphaB;

  float* _sumsFBData;

  int _threads;

public:
  Parallel_For_mergeLocalHistograms(const cv::Mat& notNormalizedFG, const cv::Mat& notNormalizedBG, cv::Mat& normalizedFG, cv::Mat& normalizedBG, cv::Mat& initialized, const std::vector<cv::Point3i> centersIds, const cv::Mat& sumsFB, float alphaF, float alphaB, int threads)
  {
    histogramSize = notNormalizedFG.cols;

    notNormalizedFGData = (float*)notNormalizedFG.ptr<float>();
    notNormalizedBGData = (float*)notNormalizedBG.ptr<float>();

    normalizedFGData = (float*)normalizedFG.ptr<float>();
    normalizedBGData = (float*)normalizedBG.ptr<float>();

    initializedData = initialized.data;

    _centersIds = centersIds;

    _sumsFB = sumsFB;

    _alphaF = alphaF;
    _alphaB = alphaB;

    _sumsFBData = (float*)_sumsFB.ptr<float>();

    _threads = threads;
  }

  virtual void operator()(const cv::Range& r) const
  {
    //int range = _sumsFB.rows / _threads;
    int range = (int)_centersIds.size() / _threads;

    int hEnd = r.end * range;
    if (r.end == _threads)
    {
      //hEnd = _sumsFB.rows;
      hEnd = (int)_centersIds.size();
    }

    for (int h = r.start * range; h < hEnd; h++)
    {
      int cID = _centersIds[h].z;

      float* notNormalizedFG = notNormalizedFGData + h * histogramSize;
      float* notNormalizedBG = notNormalizedBGData + h * histogramSize;

      float* normalizedFG = normalizedFGData + cID * histogramSize;
      float* normalizedBG = normalizedBGData + cID * histogramSize;

      float totalFGPixels = _sumsFBData[cID * 2];
      float totalBGPixels = _sumsFBData[cID * 2 + 1];

      if (initializedData[cID] == 0)
      {
        for (int i = 0; i < histogramSize; i++)
        {
          if (false)
          {
            normalizedFG[i] = notNormalizedFG[i] / totalFGPixels;
            normalizedBG[i] = notNormalizedBG[i] / totalBGPixels;

          }
          else
          {
            if (notNormalizedFG[i])
            {
              normalizedFG[i] = (float)notNormalizedFG[i] / totalFGPixels;
            }
            if (notNormalizedBG[i])
            {
              normalizedBG[i] = (float)notNormalizedBG[i] / totalBGPixels;
            }
          }
        }
        initializedData[cID] = 1;
      }
      else
      {
        for (int i = 0; i < histogramSize; i++)
        {

          if (false)
          {
            normalizedFG[i] = notNormalizedFG[i] / totalFGPixels;
            normalizedBG[i] = notNormalizedBG[i] / totalBGPixels;
          }
          else
          {
            if (notNormalizedFG[i])
            {
              normalizedFG[i] = (1.0f - _alphaF) * normalizedFG[i] + _alphaF * (float)notNormalizedFG[i] / totalFGPixels;
            }
            if (notNormalizedBG[i])
            {
              normalizedBG[i] = (1.0f - _alphaB) * normalizedBG[i] + _alphaB * (float)notNormalizedBG[i] / totalBGPixels;
            }
          }
        }

      }
    }
  }
};

/**
 *  This class extends the OpenCV ParallelLoopBody for efficiently parallelized
 *  computations. Within the corresponding for loop, every 3D histogram center is projected
 *  into the image plane. Those that do not project on or close to the object's contour are
 *  being filtered based on a given binary silhouette mask and depth map at a specified image
 *  pyramid level.
 */
class Parallel_For_computeHistogramCenters : public cv::ParallelLoopBody
{
private:
  std::vector<cv::Vec3f> _verticies;

  std::vector<cv::Point3i>* _centersIds;

  cv::Mat _depth;
  cv::Mat _mask;

  uchar* maskData;

  cv::Matx44f _T_cm;
  cv::Matx33f _K;

  float _zNear;
  float _zFar;

  int _m_id;

  int _level;

  int downScale;
  int upScale;

  int _threads;

public:
  Parallel_For_computeHistogramCenters(const cv::Mat& mask, const cv::Mat& depth, const std::vector<cv::Vec3f>& verticies, const cv::Matx44f& T_cm, const cv::Matx33f& K, float zNear, float zFar, int m_id, int level, std::vector<cv::Point3i>* centersIds, int threads)
  {
    _verticies = verticies;

    _depth = depth;

    _level = level;

    downScale = pow(2, 2 - level);

    upScale = pow(2, level);

    if (mask.type() % 8 == 5)
    {
      mask.convertTo(_mask, CV_8UC1, 10000);
    }
    else
    {
      _mask = mask;
    }

    maskData = mask.data;

    _T_cm = T_cm;
    _K = K;

    _zNear = zNear;
    _zFar = zFar;

    _m_id = m_id;

    _centersIds = centersIds;

    _threads = threads;
  }

  virtual void operator()(const cv::Range& r) const
  {
    int range = (int)_verticies.size() / _threads;

    int vEnd = r.end * range;
    if (r.end == _threads)
    {
      vEnd = (int)_verticies.size();
    }

    std::vector<cv::Point3i>* tmp = &_centersIds[r.start];

    for (int v = r.start * range; v < vEnd; v++)
    {
      cv::Vec3f V_m = _verticies[v];

      float X_m = V_m[0];
      float Y_m = V_m[1];
      float Z_m = V_m[2];

      float X_c = X_m * _T_cm(0, 0) + Y_m * _T_cm(0, 1) + Z_m * _T_cm(0, 2) + _T_cm(0, 3);
      float Y_c = X_m * _T_cm(1, 0) + Y_m * _T_cm(1, 1) + Z_m * _T_cm(1, 2) + _T_cm(1, 3);
      float Z_c = X_m * _T_cm(2, 0) + Y_m * _T_cm(2, 1) + Z_m * _T_cm(2, 2) + _T_cm(2, 3);

      float x = X_c / Z_c * _K(0, 0) + _K(0, 2);
      float y = Y_c / Z_c * _K(1, 1) + _K(1, 2);

      if (x >= 0 && x < _depth.cols && y >= 0 && y < _depth.rows)
      {
        float d = 1.0f - _depth.at<float>(y, x);

        float Z_d = 2.0f * _zNear * _zFar / (_zFar + _zNear - (2.0f * (d)-1.0) * (_zFar - _zNear));

        if (fabs(Z_c - Z_d) < 1.0f || d == 1.0)
        {
          int xi = (int)x;
          int yi = (int)y;

          if (xi >= downScale && xi < _mask.cols - downScale && yi >= downScale && yi < _mask.rows - downScale)
          {
            uchar v0 = maskData[yi * _mask.cols + xi] == _m_id;
            uchar v1 = maskData[yi * _mask.cols + xi + downScale] == _m_id;
            uchar v2 = maskData[yi * _mask.cols + xi - downScale] == _m_id;
            uchar v3 = maskData[(yi + downScale) * _mask.cols + xi] == _m_id;
            uchar v4 = maskData[(yi - downScale) * _mask.cols + xi] == _m_id;

            if (v0 * v1 * v2 * v3 * v4 == 0)
            {
              tmp->push_back(cv::Point3i(x * upScale, y * upScale, v));
            }
          }
        }
      }
    }
  }
};

void TCLCHistograms::TestLine(uchar* frameRow, uchar* maskRow, int xl, int xr, float* localHistogramFG, float* localHistogramBG, float& sum_err, float& sum_all)
{
  uchar* frame_ptr = (uchar*)(frameRow)+3 * xl;

  uchar* mask_ptr = (uchar*)(maskRow)+xl;
  uchar* mask_max_ptr = (uchar*)(maskRow)+xr;
  int _m_id = _model->getModelID();
  int _binShift = 8 - log(numBins) / log(2);

  for (; mask_ptr <= mask_max_ptr; mask_ptr += 1, frame_ptr += 3)
  {
    sum_all += 1;

    int pidx;
    int ru, gu, bu;

    ru = (frame_ptr[0] >> _binShift);
    gu = (frame_ptr[1] >> _binShift);
    bu = (frame_ptr[2] >> _binShift);
    pidx = (ru * numBins + gu) * numBins + bu;
      
    float pf = localHistogramFG[pidx];
		float pb = localHistogramBG[pidx];

		pf += 0.0000001f;
		pb += 0.0000001f;

		//ppf += etaf*pf / (etaf*pf + etab*pb);
		//ppb += etab*pb / (etaf*pf + etab*pb);

		float ppf = pf / (pf + pb);
		float ppb = pb / (pf + pb);

    if ((*mask_ptr == _m_id && ppf < ppb) || (*mask_ptr != _m_id && ppf > ppb)) {
      sum_err += 1;
    }
  }
}

float TCLCHistograms::ComputeWeight(const cv::Mat& frame, const cv::Mat& mask, cv::Point3i& center, int radius) {
	uchar* frame_data = frame.data;
	uchar* mask_data = mask.data;

	size_t frame_step = frame.step;
	size_t mask_step = mask.step;

	int err = 0;
	int dx = radius;
	int dy = 0;
	int plus = 1;
	int minus = (radius << 1) - 1;
	int olddx = dx;
		
	int inside = center.x >= radius && center.x < frame.cols - radius && center.y >= radius && center.y < frame.rows - radius;

  cv::Mat localFG = getLocalForegroundHistograms();
	cv::Mat localBG = getLocalBackgroundHistograms();
	float* localFGData = (float*)localFG.ptr<float>();
	float* localBGData = (float*)localBG.ptr<float>();
  int histogramSize = localFG.cols;

  float* localHistogramFG = localFGData + center.z * histogramSize;
  float* localHistogramBG = localBGData + center.z * histogramSize;
  float sum_err = 0, sum_all = 0;

	while (dx >= dy) {
		int mask;
		int y11 = center.y - dy, y12 = center.y + dy, y21 = center.y - dx, y22 = center.y + dx;
		int x11 = center.x - dx, x12 = center.x + dx, x21 = center.x - dy, x22 = center.x + dy;

		if (inside) {
			uchar* frame_row = frame_data + y11 * frame_step;
			uchar* mask_row = mask_data + y11 * mask_step;
			TestLine(frame_row, mask_row, x11, x12, localHistogramFG, localHistogramBG, sum_err, sum_all);

			if (y11 != y12) {
				uchar* frame_row = frame_data + y12 * frame_step;
				uchar* mask_row = mask_data + y12 * mask_step;
				TestLine(frame_row, mask_row, x11, x12, localHistogramFG, localHistogramBG, sum_err, sum_all);
			}

			if (olddx != dx) {
				if (y11 != y21) {
					uchar* frame_row = frame_data + y21 * frame_step;
					uchar* mask_row = mask_data + y21 * mask_step;
					TestLine(frame_row, mask_row, x21, x22, localHistogramFG, localHistogramBG, sum_err, sum_all);
				}

				if (y12 != y22) {
					uchar* frame_row = frame_data + y22 * frame_step;
					uchar* mask_row = mask_data + y22 * mask_step;
					TestLine(frame_row, mask_row, x21, x22, localHistogramFG, localHistogramBG, sum_err, sum_all);
				}
			}
		}	else if (x11 < frame.cols && x12 >= 0 && y21 < frame.rows && y22 >= 0) {
			x11 = std::max(x11, 0);
			x12 = MIN(x12, frame.cols - 1);

			if ((unsigned)y11 < (unsigned)frame.rows) {
				uchar* frame_row = frame_data + y11 * frame_step;
				uchar* mask_row = mask_data + y11 * mask_step;
				TestLine(frame_row, mask_row, x11, x12, localHistogramFG, localHistogramBG, sum_err, sum_all);
			}

			if ((unsigned)y12 < (unsigned)frame.rows && (y11 != y12)) {
				uchar* frame_row = frame_data + y12 * frame_step;
				uchar* mask_row = mask_data + y12 * mask_step;
				TestLine(frame_row, mask_row, x11, x12, localHistogramFG, localHistogramBG, sum_err, sum_all);
			}

			if (x21 < frame.cols && x22 >= 0 && (olddx != dx)) {
				x21 = std::max(x21, 0);
				x22 = MIN(x22, frame.cols - 1);

				if ((unsigned)y21 < (unsigned)frame.rows) {
					uchar* frame_row = frame_data + y21 * frame_step;
					uchar* mask_row = mask_data + y21 * mask_step;
					TestLine(frame_row, mask_row, x21, x22, localHistogramFG, localHistogramBG, sum_err, sum_all);
				}

				if ((unsigned)y22 < (unsigned)frame.rows) {
					uchar* frame_row = frame_data + y22 * frame_step;
					uchar* mask_row = mask_data + y22 * mask_step;
					TestLine(frame_row, mask_row, x21, x22, localHistogramFG, localHistogramBG, sum_err, sum_all);
				}
			}
		}

		olddx = dx;

		dy++;
		err += plus;
		plus += 2;

		mask = (err <= 0) - 1;

		err -= minus & mask;
		dx += mask;
		minus -= mask & 2;
	}

	return (sum_all - sum_err) / sum_all;
}

void TCLCHistograms::update(const Mat &frame, const Mat &mask, const Mat &depth, Matx33f &K, float zNear, float zFar, float afg, float abg)
{
    _centersIDs = parallelComputeLocalHistogramCenters(mask, depth, K, zNear, zFar, 0);
    
    filterHistogramCenters(100, 10.0f);
    
    int threads = (int)_centersIDs.size();
    
    memset(notNormalizedFG.ptr<int>(), 0, _centersIDs.size()*numBins*numBins*numBins*sizeof(float));
    memset(notNormalizedBG.ptr<int>(), 0, _centersIDs.size()*numBins*numBins*numBins*sizeof(float));
    
    memset(normalizedFG.ptr<float>(), 0, _centersIDs.size()*numBins*numBins*numBins*sizeof(float));
    memset(normalizedBG.ptr<float>(), 0, _centersIDs.size()*numBins*numBins*numBins*sizeof(float));
    
    //Mat sumsFB = Mat::zeros((int)_centersIDs.size(), 1, CV_32SC2);
    
    parallel_for_(cv::Range(0, threads), Parallel_For_buildLocalHistograms(frame, mask, _centersIDs, radius, numBins, notNormalizedFG, notNormalizedBG, sumsFB, _model->getModelID(), threads));
    
    parallel_for_(cv::Range(0, threads), Parallel_For_mergeLocalHistograms(notNormalizedFG, notNormalizedBG, normalizedFG, normalizedBG, initialized, _centersIDs, sumsFB, afg, abg, threads));

    //wes.resize(_centersIDs.size());
    //for (int i = 0; i < _centersIDs.size(); ++i) {
    //  wes[i] = ComputeWeight(frame, mask, _centersIDs[i], radius);
    //}
}

void TCLCHistograms::updateCentersAndIds(const cv::Mat &mask, const cv::Mat &depth, const cv::Matx33f &K, float zNear, float zFar, int level)
{
    _centersIDs = parallelComputeLocalHistogramCenters(mask, depth, K, zNear, zFar, level);
    
    filterHistogramCenters(100, 10.0f);
}


vector<Point3i> TCLCHistograms::computeLocalHistogramCenters(const Mat &mask)
{
    uchar *maskData = mask.data;
    
    std::vector<Point3i> centers;
    
    int m_id = _model->getModelID();
    
    for(int i = 2; i < mask.rows - 2; i+=2)
    {
        for(int j = 2; j < mask.cols - 2; j+=2)
        {
            int idx = i*mask.cols + j;
            uchar val = maskData[idx];
            if(val == m_id)
            {
                if(maskData[idx + 2] != m_id || maskData[idx - 2] != m_id
                   || maskData[idx + 2*mask.cols] != m_id || maskData[idx - 2*mask.cols] != m_id
                   )
                {
                    centers.push_back(Point3i(j, i, (int)centers.size()));
                }
            }
        }
    }
    
    return centers;
}


vector<Point3i> TCLCHistograms::parallelComputeLocalHistogramCenters(const Mat &mask, const Mat &depth, const Matx33f &K, float zNear, float zFar, int level)
{
    vector<Point3i> res;
    
    vector<Vec3f> verticies = _model->getSimpleVertices();
    Matx44f T_cm = _model->getPose();
    Matx44f T_n = _model->getNormalization();
    
    vector<vector<Point3i> > centersIdsCollection;
    centersIdsCollection.resize(8);
    
    Matx44f T_cm_n = T_cm * T_n;
    
    int m_id = _model->getModelID();
    
    parallel_for_(cv::Range(0, 8), Parallel_For_computeHistogramCenters(mask, depth, verticies, T_cm_n, K, zNear, zFar, m_id, level, centersIdsCollection.data(), 8));
    
    for(int i = 0; i < centersIdsCollection.size(); i++)
    {
        vector<Point3i> tmp = centersIdsCollection[i];
        for(int j = 0; j < tmp.size(); j++)
        {
            res.push_back(tmp[j]);
        }
    }
    
    return res;
    
}


void TCLCHistograms::filterHistogramCenters(int numHistograms, float offset)
{
    int offset2 = (offset)*(offset);
    
    vector<Point3i> res;
    
    do
    {
        res.clear();
        
        while(_centersIDs.size() > 0)
        {
            Point3i center = _centersIDs[0];
            vector<Point3i> tmp;
            res.push_back(center);
            for(int c2 = 1; c2 < _centersIDs.size(); c2++)
            {
                Point3i center2 = _centersIDs[c2];
                int dx = center.x - center2.x;
                int dy = center.y - center2.y;
                int d = dx*dx + dy*dy;
                
                if(d >= offset2)
                {
                    tmp.push_back(center2);
                }
            }
            _centersIDs = tmp;
        }
        _centersIDs = res;
        
        offset += 1.0f;
        offset2 = offset*offset;
    }
    while(res.size() > numHistograms);
    
    _offset = offset;
}


Mat TCLCHistograms::getLocalForegroundHistograms()
{
    return normalizedFG;
}


Mat TCLCHistograms::getLocalBackgroundHistograms()
{
    return normalizedBG;
}


vector<Point3i> TCLCHistograms::getCentersAndIDs()
{
    return _centersIDs;
}


Mat TCLCHistograms::getInitialized()
{
    return initialized;
}


int TCLCHistograms::getNumBins()
{
    return numBins;
}

int TCLCHistograms::getNumHistograms()
{
    return _numHistograms;
}

int TCLCHistograms::getRadius()
{
    return radius;
}


float TCLCHistograms::getOffset()
{
    return _offset;
}


void TCLCHistograms::clear()
{
    normalizedFG = Mat::zeros(this->_numHistograms, numBins*numBins*numBins, CV_32FC1);
    normalizedBG = Mat::zeros(this->_numHistograms, numBins*numBins*numBins, CV_32FC1);
    
    notNormalizedFG = Mat::zeros(this->_numHistograms, numBins*numBins*numBins, CV_32FC1);
    notNormalizedBG = Mat::zeros(this->_numHistograms, numBins*numBins*numBins, CV_32FC1);
    
    initialized = Mat::zeros(1, this->_numHistograms, CV_8UC1);
}

class Parallel_For_buildWeightedLocalHistograms : public cv::ParallelLoopBody {
private:
  cv::Mat _frame;
  cv::Mat _mask;

  uchar* frameData;
  uchar* maskData;

  size_t frameStep;
  size_t maskStep;

  cv::Size size;

  std::vector<cv::Point3i> _centers;

  int _radius;

  int _numBins;

  int _binShift;

  int histogramSize;

  cv::Mat _sumsFB;

  int _m_id;

  float* localFGData;
  float* localBGData;
  
  float* sdt_data;
  size_t sdt_cols;

  float* _sumsFBData;

  int _threads;

public:
  Parallel_For_buildWeightedLocalHistograms(const cv::Mat& frame, const cv::Mat& mask, const cv::Mat sdt, const std::vector<cv::Point3i>& centers, float radius, int numBins, cv::Mat& localHistogramsFG, cv::Mat& localHistogramsBG, cv::Mat& sumsFB, int m_id, int threads)
  {
    _frame = frame;
    _mask = mask;

    frameData = _frame.data;
    maskData = _mask.data;

    frameStep = _frame.step;
    maskStep = _mask.step;

    size = frame.size();

    _centers = centers;

    _radius = radius;

    _numBins = numBins;

    _binShift = 8 - log(numBins) / log(2);

    histogramSize = localHistogramsFG.cols;

    localFGData = (float*)localHistogramsFG.ptr<float>();
    localBGData = (float*)localHistogramsBG.ptr<float>();

    sdt_data = (float*)sdt.ptr<float>();
    sdt_cols = sdt.cols;

    _sumsFB = sumsFB;

    _m_id = m_id;

    _sumsFBData = (float*)_sumsFB.ptr<float>();

    _threads = threads;
  }

  void processLine(uchar* frameRow, uchar* maskRow, float* sdt_row, int xl, int xr, float* localHistogramFG, float* localHistogramBG, float* sumFB) const
  {
    uchar* frame_ptr = (uchar*)(frameRow)+3 * xl;

    uchar* mask_ptr = (uchar*)(maskRow)+xl;
    uchar* mask_max_ptr = (uchar*)(maskRow)+xr;
    float* sdt_ptr = (float*)(sdt_row)+xl;
    for (; mask_ptr <= mask_max_ptr; sdt_ptr += 1, mask_ptr += 1, frame_ptr += 3)
    {
      int pidx;
      int ru, gu, bu;

      ru = (frame_ptr[0] >> _binShift);
      gu = (frame_ptr[1] >> _binShift);
      bu = (frame_ptr[2] >> _binShift);
      pidx = (ru * _numBins + gu) * _numBins + bu;
      
      //float dt = fabs(*sdt_ptr);
      //float wp = -exp(0.01 * dt);
      if (*mask_ptr == _m_id)
      {
        localHistogramFG[pidx] += 1;
        sumFB[0]++;
        //localHistogramFG[pidx] += wp;
        //sumFB[0] += wp;
      }
      else
      {
        localHistogramBG[pidx] += 1;
        sumFB[1]++;
        //localHistogramBG[pidx] += wp;
        //sumFB[1] += wp;
      }
    }
  }

  virtual void operator()(const cv::Range& r) const {
    int range = (int)_centers.size() / _threads;

    int cEnd = r.end * range;
    if (r.end == _threads) {
      cEnd = (int)_centers.size();
    }

    for (int c = r.start * range; c < cEnd; c++) {
      int err = 0;
      int dx = _radius;
      int dy = 0;
      int plus = 1;
      int minus = (_radius << 1) - 1;

      int olddx = dx;

      cv::Point3i center = _centers[c];

      int inside = center.x >= _radius && center.x < size.width - _radius && center.y >= _radius && center.y < size.height - _radius;

      float* localHistogramFG = localFGData + c * histogramSize;
      float* localHistogramBG = localBGData + c * histogramSize;

      int cID = _centers[c].z;
      float* sumFB = _sumsFBData + cID * 2;
      sumFB[0] = 0;
      sumFB[1] = 0;

      while (dx >= dy) {
        int mask;
        int y11 = center.y - dy, y12 = center.y + dy, y21 = center.y - dx, y22 = center.y + dx;
        int x11 = center.x - dx, x12 = center.x + dx, x21 = center.x - dy, x22 = center.x + dy;

        float* sdt_row;

        if (inside) {
          uchar* frameRow = frameData + y11 * frameStep;
          uchar* maskRow = maskData + y11 * maskStep;
          float* sdt_row = sdt_data + y11 * sdt_cols;
          processLine(frameRow, maskRow, sdt_row, x11, x12, localHistogramFG, localHistogramBG, sumFB);
          
          if (y11 != y12) {
            uchar* frameRow = frameData + y12 * frameStep;
            uchar* maskRow = maskData + y12 * maskStep;
            float* sdt_row = sdt_data + y12 * sdt_cols;
            processLine(frameRow, maskRow, sdt_row, x11, x12, localHistogramFG, localHistogramBG, sumFB);
          }

          if (olddx != dx) {
            if (y11 != y21) {
              frameRow = frameData + y21 * frameStep;
              maskRow = maskData + y21 * maskStep;
              float* sdt_row = sdt_data + y21 * sdt_cols;
              processLine(frameRow, maskRow, sdt_row, x21, x22, localHistogramFG, localHistogramBG, sumFB);
            }
            
            if (y12 != y22) {
              uchar* frameRow = frameData + y22 * frameStep;
              uchar* maskRow = maskData + y22 * maskStep;
              float* sdt_row = sdt_data + y22 * sdt_cols;
              processLine(frameRow, maskRow, sdt_row, x21, x22, localHistogramFG, localHistogramBG, sumFB);
            }
          }
        } else if (x11 < size.width && x12 >= 0 && y21 < size.height && y22 >= 0) {
          x11 = std::max(x11, 0);
          x12 = MIN(x12, size.width - 1);

          if ((unsigned)y11 < (unsigned)size.height) {
            uchar* frameRow = frameData + y11 * frameStep;
            uchar* maskRow = maskData + y11 * maskStep;
            float* sdt_row = sdt_data + y11 * sdt_cols;
            processLine(frameRow, maskRow, sdt_row, x11, x12, localHistogramFG, localHistogramBG, sumFB);
          }

          if ((unsigned)y12 < (unsigned)size.height && (y11 != y12)) {
            uchar* frameRow = frameData + y12 * frameStep;
            uchar* maskRow = maskData + y12 * maskStep;
            float* sdt_row = sdt_data + y12 * sdt_cols;
            processLine(frameRow, maskRow, sdt_row, x11, x12, localHistogramFG, localHistogramBG, sumFB);
          }

          if (x21 < size.width && x22 >= 0 && (olddx != dx)) {
            x21 = std::max(x21, 0);
            x22 = MIN(x22, size.width - 1);

            if ((unsigned)y21 < (unsigned)size.height) {
              uchar* frameRow = frameData + y21 * frameStep;
              uchar* maskRow = maskData + y21 * maskStep;
              float* sdt_row = sdt_data + y21 * sdt_cols;
              processLine(frameRow, maskRow, sdt_row, x21, x22, localHistogramFG, localHistogramBG, sumFB);
            }

            if ((unsigned)y22 < (unsigned)size.height) {
              uchar* frameRow = frameData + y22 * frameStep;
              uchar* maskRow = maskData + y22 * maskStep;
              float* sdt_row = sdt_data + y22 * sdt_cols;
              processLine(frameRow, maskRow, sdt_row, x21, x22, localHistogramFG, localHistogramBG, sumFB);
            }
          }
        }

        olddx = dx;

        dy++;
        err += plus;
        plus += 2;

        mask = (err <= 0) - 1;

        err -= minus & mask;
        dx += mask;
        minus -= mask & 2;
      }
    }
  }
};

WTCLCHistograms::WTCLCHistograms(Model* model, int numBins, int radius, float offset)
  : TCLCHistograms(model, numBins, radius, offset)
{
  SDT2D = new SignedDistanceTransform2D(8.0f);
}

WTCLCHistograms::~WTCLCHistograms()
{

}

void WTCLCHistograms::update(const Mat& frame, const Mat& mask, const Mat& depth, Matx33f& K, float zNear, float zFar, float afg, float abg) {
  _centersIDs = parallelComputeLocalHistogramCenters(mask, depth, K, zNear, zFar, 0);

  filterHistogramCenters(100, 10.0f);

  int threads = (int)_centersIDs.size();

  memset(notNormalizedFG.ptr<int>(), 0, _centersIDs.size() * numBins * numBins * numBins * sizeof(float));
  memset(notNormalizedBG.ptr<int>(), 0, _centersIDs.size() * numBins * numBins * numBins * sizeof(float));

  memset(normalizedFG.ptr<float>(), 0, _centersIDs.size() * numBins * numBins * numBins * sizeof(float));
  memset(normalizedBG.ptr<float>(), 0, _centersIDs.size() * numBins * numBins * numBins * sizeof(float));

  //Mat sumsFB = Mat::zeros((int)_centersIDs.size(), 1, CV_32SC2);
  
  cv::Mat sdt, xyPos;
  SDT2D->computeTransform(mask, sdt, xyPos, 8, _model->getModelID());
  //cv::imshow("sdt", sdt);
  //cv::waitKey();
  parallel_for_(cv::Range(0, threads), Parallel_For_buildWeightedLocalHistograms(frame, mask, sdt, _centersIDs, radius, numBins, notNormalizedFG, notNormalizedBG, sumsFB, _model->getModelID(), threads));

  parallel_for_(cv::Range(0, threads), Parallel_For_mergeLocalHistograms(notNormalizedFG, notNormalizedBG, normalizedFG, normalizedBG, initialized, _centersIDs, sumsFB, afg, abg, threads));
}