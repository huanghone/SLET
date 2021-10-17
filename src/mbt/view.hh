#pragma once


#include <iostream>
#include <vector>

#include <QOpenGLContext>
#include <QOffscreenSurface>

#include <QGLFramebufferObject>
#include <QOpenGLShaderProgram>
#include <QOpenGLFunctions_3_3_Core>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "transformations.hh"
#include "model.hh"

class View : public QOpenGLFunctions_3_3_Core {
public:
	enum FrameType {
		MASK,
		RGB,
		RGB_32F,
		DEPTH
	};

	View(void);

	~View(void);

	static View* Instance(void) {
		if (instance == NULL) instance = new View();
		return instance;
	}

	bool IsInFrame(cv::Point2f& pt) {
		return (pt.x >= 0 && pt.x < width && pt.y >= 0 && pt.y < height);
	}

	void init(const cv::Matx33f& K, int width, int height, float zNear, float zFar, int numLevels);

	int getNumLevels();
	void setLevel(int level);
	int getLevel();

	float getZNear();
	float getZFar();

	int GetWidth() { return fullWidth; }
	int GetHeight() { return fullHeight; }
	cv::Matx44f GetCalibrationMatrix();
	
	void Project(const cv::Matx44f& mv_mat, std::vector<cv::Vec3f>& model_points, std::vector<cv::Vec2f>& image_points);
	void RenderCV(Model* model, cv::Mat& buf);
	void RenderCV(Model* model, cv::Mat& buf, cv::Scalar color);

	void RenderSilhouette(Model* model, GLenum polyonMode, bool invertDepth = false, float r = 1.0f, float g = 1.0f, float b = 1.0f, bool drawAll = false);
	void RenderSilhouette(std::vector<Model*> models, GLenum polyonMode, bool invertDepth = false, const std::vector<cv::Point3f>& colors = std::vector<cv::Point3f>(), bool drawAll = false);

	void RenderShaded(Model* model, GLenum polyonMode, float r = 1.0f, float g = 0.5f, float b = 0.0f, bool drawAll = false);
	void RenderShaded(std::vector<Model*> models, GLenum polyonMode, const std::vector<cv::Point3f>& colors = std::vector<cv::Point3f>(), bool drawAll = false);

	void RenderNormals(Model* model, GLenum polyonMode, bool drawAll = false);
	void RenderNormals(std::vector<Model*> models, GLenum polyonMode, bool drawAll = false);

	void ConvertMask(const cv::Mat& src_mask, cv::Mat& mask, uchar oid);

	void ProjectBoundingBox(Model* model, std::vector<cv::Point2f>& projections, cv::Matx44f& pose, cv::Rect& boundingRect);
	void ProjectBoundingBox(Model* model, std::vector<cv::Point2f>& projections, cv::Rect& boundingRect);
	//void BackProject(cv::Point& pt, cv::Point3f& pt3d);
	void ProjectPoints(const std::vector<cv::Point3f>& pts3d, const cv::Matx44f& pose, std::vector<cv::Point2f>& pts);
	void ProjectPoints(const std::vector<cv::Point3f>& pts3d, const cv::Matx44f& pose, std::vector<cv::Point>& pts);
	void BackProjectPoints(std::vector<cv::Point>& pts, const cv::Mat& depth_map, const cv::Matx44f& pose, std::vector<cv::Point3f>& pts3d);
	
	cv::Mat DownloadFrame(View::FrameType type);

	void destroy();

protected:
	QOpenGLContext* getContext();
	GLuint getFrameBufferID();
	GLuint getColorTextureID();
	GLuint getDepthTextureID();

	void makeCurrent();
	void doneCurrent();

private:
	static View* instance;

	int width;
	int height;

	int fullWidth;
	int fullHeight;

	float zn;
	float zf;

	int numLevels;

	int currentLevel;

	std::vector<cv::Matx44f> calibrationMatrices;
	cv::Matx44f projectionMatrix;
	cv::Matx44f lookAtMatrix;

	QOffscreenSurface* surface;
	QOpenGLContext* glContext;

	GLuint frameBufferID;
	GLuint colorTextureID;
	GLuint depthTextureID;

	int angle;

	cv::Vec3f lightPosition;

	QString shaderFolder;
	QOpenGLShaderProgram* silhouetteShaderProgram;
	QOpenGLShaderProgram* phongblinnShaderProgram;
	QOpenGLShaderProgram* normalsShaderProgram;

	bool initRenderingBuffers();
	bool initShaderProgramFromCode(QOpenGLShaderProgram* program, char* vertex_shader, char* fragment_shader);
};
