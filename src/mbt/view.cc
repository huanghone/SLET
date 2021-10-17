#include <iostream>

#include <glog/logging.h>
#include "view.hh"

using namespace std;
using namespace cv;

View* View::instance;

View::View(void) {
	QSurfaceFormat glFormat;
	glFormat.setVersion(3, 3);
	glFormat.setProfile(QSurfaceFormat::CoreProfile);
	glFormat.setRenderableType(QSurfaceFormat::OpenGL);

	surface = new QOffscreenSurface();
	surface->setFormat(glFormat);
	surface->create();

	glContext = new QOpenGLContext();
	glContext->setFormat(surface->requestedFormat());
	glContext->create();

	silhouetteShaderProgram = new QOpenGLShaderProgram();
	phongblinnShaderProgram = new QOpenGLShaderProgram();
	normalsShaderProgram = new QOpenGLShaderProgram();

	calibrationMatrices.push_back(Matx44f::eye());

	projectionMatrix = Transformations::perspectiveMatrix(40, 4.0f / 3.0f, 0.1, 1000.0);

	lookAtMatrix = Transformations::lookAtMatrix(0, 0, 0, 0, 0, 1, 0, -1, 0);

	currentLevel = 0;
}

View::~View(void) {
	glDeleteTextures(1, &colorTextureID);
	glDeleteTextures(1, &depthTextureID);
	glDeleteFramebuffers(1, &frameBufferID);

	delete phongblinnShaderProgram;
	delete normalsShaderProgram;
	delete silhouetteShaderProgram;
	delete surface;
}

void View::destroy() {
	doneCurrent();

	glBindTexture(GL_TEXTURE_2D, 0);
	glBindRenderbuffer(GL_RENDERBUFFER, 0);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	delete instance;
	instance = NULL;
}

void View::makeCurrent() {
	glContext->makeCurrent(surface);
}

void View::doneCurrent() {
	glContext->doneCurrent();
}

QOpenGLContext* View::getContext() {
	return glContext;
}


GLuint View::getFrameBufferID() {
	return frameBufferID;
}


GLuint View::getColorTextureID() {
	return colorTextureID;
}


GLuint View::getDepthTextureID() {
	return depthTextureID;
}

float View::getZNear() {
	return zn;
}

float View::getZFar() {
	return zf;
}

Matx44f View::GetCalibrationMatrix() {
	return calibrationMatrices[currentLevel];
}

#include "shader/shaders.hh"

void View::init(const Matx33f& K, int width, int height, float zNear, float zFar, int numLevels) {
	this->width = width;
	this->height = height;

	fullWidth = width;
	fullHeight = height;

	this->zn = zNear;
	this->zf = zFar;

	this->numLevels = numLevels;

	projectionMatrix = Transformations::perspectiveMatrix(K, width, height, zNear, zFar, true);

	makeCurrent();

	initializeOpenGLFunctions();

	//FIX FOR NEW OPENGL
	uint vao;
	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);

	calibrationMatrices.clear();

	for (int i = 0; i < numLevels; i++) {
		float s = pow(2, i);

		Matx44f K_l = Matx44f::eye();
		K_l(0, 0) = K(0, 0) / s;
		K_l(1, 1) = K(1, 1) / s;
		K_l(0, 2) = K(0, 2) / s;
		K_l(1, 2) = K(1, 2) / s;

		calibrationMatrices.push_back(K_l);
	}

	//cout << "GL Version " << glGetString(GL_VERSION) << endl << "GLSL Version " << glGetString(GL_SHADING_LANGUAGE_VERSION) << endl;

	glEnable(GL_DEPTH);
	glEnable(GL_DEPTH_TEST);

	//INVERT DEPTH BUFFER
	glDepthRange(1, 0);
	glClearDepth(0.0f);
	glDepthFunc(GL_GREATER);

	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

	glClearColor(0.0, 0.0, 0.0, 1.0);

	initRenderingBuffers();

	shaderFolder = "src/";

	initShaderProgramFromCode(silhouetteShaderProgram, silhouette_vertex_shader, silhouette_fragment_shader);
	initShaderProgramFromCode(phongblinnShaderProgram, phongblinn_vertex_shader, phongblinn_fragment_shader);
	initShaderProgramFromCode(normalsShaderProgram, normals_vertex_shader, normals_fragment_shader);

	//initShaderProgram(silhouetteShaderProgram, "silhouette");
	//initShaderProgram(phongblinnShaderProgram, "phongblinn");
	//initShaderProgram(normalsShaderProgram, "normals");

	angle = 0;

	lightPosition = cv::Vec3f(0, 0, 0);

	//doneCurrent();
}

int View::getNumLevels() {
	return numLevels;
}

void View::setLevel(int level) {
	currentLevel = level;
	int s = pow(2, currentLevel);
	width = fullWidth / s;
	height = fullHeight / s;

	width += width % 4;
	height += height % 4;
}


int View::getLevel() {
	return currentLevel;
}


bool View::initRenderingBuffers() {
	glGenTextures(1, &colorTextureID);
	glBindTexture(GL_TEXTURE_2D, colorTextureID);

	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

	glGenTextures(1, &depthTextureID);
	glBindTexture(GL_TEXTURE_2D, depthTextureID);

	glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, width, height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

	glGenFramebuffers(1, &frameBufferID);
	glBindFramebuffer(GL_FRAMEBUFFER, frameBufferID);

	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, colorTextureID, 0);

	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depthTextureID, 0);

	if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
		cout << "error creating rendering buffers" << endl;
		return false;
	}
	return true;
}

bool View::initShaderProgramFromCode(QOpenGLShaderProgram* program, char* vertex_shader, char* fragment_shader) {
	if (!program->addShaderFromSourceCode(QOpenGLShader::Vertex, vertex_shader)) {
		cout << "error adding vertex shader from source file" << endl;
		return false;
	}
	if (!program->addShaderFromSourceCode(QOpenGLShader::Fragment, fragment_shader)) {
		cout << "error adding fragment shader from source file" << endl;
		return false;
	}

	if (!program->link()) {
		cout << "error linking shaders" << endl;
		return false;
	}
	return true;
}

void View::Project(const cv::Matx44f& mv_mat, std::vector<cv::Vec3f>& model_points, std::vector<cv::Vec2f>& image_points) {
	for (int i = 0; i < model_points.size(); ++i) {
		cv::Matx44f kmat = View::GetCalibrationMatrix();
		cv::Vec4f ptv = kmat*mv_mat*cv::Vec4f(model_points[i](0), model_points[i](1), model_points[i](2), 1);

		float dz = 1.0f/ptv(2);
		image_points[i](0) = ptv(0)*dz;
		image_points[i](1) = ptv(1)*dz;
	}
}

static bool PtInFrame(const cv::Vec2f& pt, int width, int height) {
	return (pt(0) < width && pt(1) < height && pt(0) >= 0 && pt(1) >= 0);
}

void View::RenderCV(Model* model, cv::Mat& frame, cv::Scalar color) {
	std::vector<cv::Vec3f>& model_points = model->vertices;
	std::vector<GLuint>& indices = model->indices;

	std::vector<cv::Vec2f> image_points(model_points.size());
	Project(model->getPose(), model_points, image_points);

	cv::Mat buf(frame.size(), CV_8UC3, cv::Scalar(0,0,0));
	for (int i = 0; i < indices.size(); i += 3) {
		cv::Vec2f pt0 = image_points[indices[i]];
		cv::Vec2f pt1 = image_points[indices[i+1]];
		cv::Vec2f pt2 = image_points[indices[i+2]];
		
		cv::line(buf, cv::Point(pt0(0), pt0(1)), cv::Point(pt1(0), pt1(1)), color, 2);
		cv::line(buf, cv::Point(pt1(0), pt1(1)), cv::Point(pt2(0), pt2(1)), color, 2);
		cv::line(buf, cv::Point(pt2(0), pt2(1)), cv::Point(pt0(0), pt0(1)), color, 2);
	}

	for (int r = 0; r < buf.rows; ++r)
	for (int c = 0; c < buf.cols; ++c) {
		float alpha = 0.5f;
		if (buf.at<cv::Vec3b>(r, c)[0]) {
			cv::Vec3b& vf = frame.at<cv::Vec3b>(r, c);
			cv::Vec3b& vb = buf.at<cv::Vec3b>(r, c);
			frame.at<cv::Vec3b>(r, c) = (1.0f - alpha) * vb + alpha * vf;
		}
	}
}

void View::RenderCV(Model* model, cv::Mat& frame) {
	std::vector<cv::Vec3f>& model_points = model->vertices;
	std::vector<GLuint>& indices = model->indices;

	std::vector<cv::Vec2f> image_points(model_points.size());
	Project(model->getPose(), model_points, image_points);

	cv::Mat buf(frame.size(), CV_8UC3, cv::Scalar(0,0,0));
	for (int i = 0; i < indices.size(); i += 3) {
		cv::Vec2f pt0 = image_points[indices[i]];
		cv::Vec2f pt1 = image_points[indices[i+1]];
		cv::Vec2f pt2 = image_points[indices[i+2]];
		
		int line_width = 1;

		cv::line(buf, cv::Point(pt0(0), pt0(1)), cv::Point(pt1(0), pt1(1)), cv::Scalar(1, 255, 1), line_width);
		cv::line(buf, cv::Point(pt1(0), pt1(1)), cv::Point(pt2(0), pt2(1)), cv::Scalar(1, 255, 1), line_width);
		cv::line(buf, cv::Point(pt2(0), pt2(1)), cv::Point(pt0(0), pt0(1)), cv::Scalar(1, 255, 1), line_width);
	}

	for (int r = 0; r < buf.rows; ++r)
	for (int c = 0; c < buf.cols; ++c) {
		float alpha = 0.5f;
		if (buf.at<cv::Vec3b>(r, c)[0]) {
			cv::Vec3b& vf = frame.at<cv::Vec3b>(r, c);
			cv::Vec3b& vb = buf.at<cv::Vec3b>(r, c);
			frame.at<cv::Vec3b>(r, c) = (1.0f - alpha) * vb + alpha * vf;
		}
	}
}

void View::RenderSilhouette(Model* model, GLenum polyonMode, bool invertDepth, float r, float g, float b, bool drawAll) {
	vector<Model*> models;
	models.push_back(model);

	vector<Point3f> colors;
	colors.push_back(Point3f(r, g, b));

	RenderSilhouette(models, polyonMode, invertDepth, colors, drawAll);
}

void View::ConvertMask(const cv::Mat& src_mask, cv::Mat& mask, uchar oid) {
	mask = cv::Mat(src_mask.size(), CV_8UC1, cv::Scalar(0));
	uchar depth = src_mask.type() & CV_MAT_DEPTH_MASK;

	if (CV_8U == depth && oid > 0) {
		for (int r = 0; r < src_mask.rows; ++r)
		for (int c = 0; c < src_mask.cols; ++c) {
			if (oid == src_mask.at<uchar>(r,c))
				mask.at<uchar>(r,c) = 255;
		}
	} else 
	if (CV_32F == depth) {
		for (int r = 0; r < src_mask.rows; ++r)
		for (int c = 0; c < src_mask.cols; ++c) {
			if (src_mask.at<float>(r,c))
				mask.at<uchar>(r,c) = 255;
		}
	}	else {
		LOG(ERROR) << "WRONG IMAGE TYPE";
	}
}

void View::RenderShaded(Model* model, GLenum polyonMode, float r, float g, float b, bool drawAll) {
	vector<Model*> models;
	models.push_back(model);

	vector<Point3f> colors;
	colors.push_back(Point3f(r, g, b));

	RenderShaded(models, polyonMode, colors, drawAll);
}

void View::RenderNormals(Model* model, GLenum polyonMode, bool drawAll) {
	vector<Model*> models;
	models.push_back(model);

	RenderNormals(models, polyonMode, drawAll);
}

void View::RenderSilhouette(vector<Model*> models, GLenum polyonMode, bool invertDepth, const std::vector<cv::Point3f>& colors, bool drawAll) {
	glViewport(0, 0, width, height);

	if (invertDepth) {
		glClearDepth(1.0f);
		glDepthFunc(GL_LESS);
	}

	glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);

	for (int i = 0; i < models.size(); i++) {
		Model* model = models[i];

		if (model->isInitialized() || drawAll) {
			Matx44f pose = model->getPose();
			Matx44f normalization = model->getNormalization();

			Matx44f modelViewMatrix = lookAtMatrix * (pose * normalization);

			Matx44f modelViewProjectionMatrix = projectionMatrix * modelViewMatrix;

			silhouetteShaderProgram->bind();
			silhouetteShaderProgram->setUniformValue("uMVPMatrix", QMatrix4x4(modelViewProjectionMatrix.val));
			silhouetteShaderProgram->setUniformValue("uAlpha", 1.0f);

			Point3f color;
			if (i < colors.size()) {
				color = colors[i];
			}
			else {
				color = Point3f((float)(model->getModelID()) / 255.0f, 0.0f, 0.0f);
			}
			silhouetteShaderProgram->setUniformValue("uColor", QVector3D(color.x, color.y, color.z));

			glPolygonMode(GL_FRONT_AND_BACK, polyonMode);

			model->draw(silhouetteShaderProgram);
		}
	}

	glClearDepth(0.0f);
	glDepthFunc(GL_GREATER);

	glFinish();
}


void View::RenderShaded(vector<Model*> models, GLenum polyonMode, const std::vector<cv::Point3f>& colors, bool drawAll) {
	glViewport(0, 0, width, height);

	glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);

	for (int i = 0; i < models.size(); i++) {
		Model* model = models[i];

		if (model->isInitialized() || drawAll) {
			Matx44f pose = model->getPose();
			Matx44f normalization = model->getNormalization();

			Matx44f modelViewMatrix = lookAtMatrix * (pose * normalization);

			Matx33f normalMatrix = modelViewMatrix.get_minor<3, 3>(0, 0).inv().t();

			Matx44f modelViewProjectionMatrix = projectionMatrix * modelViewMatrix;

			phongblinnShaderProgram->bind();
			phongblinnShaderProgram->setUniformValue("uMVMatrix", QMatrix4x4(modelViewMatrix.val));
			phongblinnShaderProgram->setUniformValue("uMVPMatrix", QMatrix4x4(modelViewProjectionMatrix.val));
			phongblinnShaderProgram->setUniformValue("uNormalMatrix", QMatrix3x3(normalMatrix.val));
			phongblinnShaderProgram->setUniformValue("uLightPosition1", QVector3D(0.1, 0.1, -0.02));
			phongblinnShaderProgram->setUniformValue("uLightPosition2", QVector3D(-0.1, 0.1, -0.02));
			phongblinnShaderProgram->setUniformValue("uLightPosition3", QVector3D(0.0, 0.0, 0.1));
			phongblinnShaderProgram->setUniformValue("uShininess", 100.0f);
			phongblinnShaderProgram->setUniformValue("uAlpha", 1.0f);

			Point3f color;
			if (i < colors.size()) {
				color = colors[i];
			}
			else {
				color = Point3f(1.0, 0.5, 0.0);
			}
			phongblinnShaderProgram->setUniformValue("uColor", QVector3D(color.x, color.y, color.z));

			glPolygonMode(GL_FRONT_AND_BACK, polyonMode);

			model->draw(phongblinnShaderProgram);
		}
	}

	glFinish();
}

void View::RenderNormals(vector<Model*> models, GLenum polyonMode, bool drawAll) {
	glViewport(0, 0, width, height);

	glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);

	for (int i = 0; i < models.size(); i++) {
		Model* model = models[i];

		if (model->isInitialized() || drawAll) {
			Matx44f pose = model->getPose();
			Matx44f normalization = model->getNormalization();

			Matx44f modelViewMatrix = lookAtMatrix * (pose * normalization);

			Matx33f normalMatrix = modelViewMatrix.get_minor<3, 3>(0, 0).inv().t();

			Matx44f modelViewProjectionMatrix = projectionMatrix * modelViewMatrix;

			normalsShaderProgram->bind();
			normalsShaderProgram->setUniformValue("uMVMatrix", QMatrix4x4(modelViewMatrix.val));
			normalsShaderProgram->setUniformValue("uMVPMatrix", QMatrix4x4(modelViewProjectionMatrix.val));
			normalsShaderProgram->setUniformValue("uNormalMatrix", QMatrix3x3(normalMatrix.val));
			normalsShaderProgram->setUniformValue("uAlpha", 1.0f);

			glPolygonMode(GL_FRONT_AND_BACK, polyonMode);

			model->draw(normalsShaderProgram);
		}
	}

	glFinish();
}

void View::ProjectBoundingBox(Model* model, std::vector<cv::Point2f>& projections, cv::Matx44f& pose, cv::Rect& boundingRect) {
	Vec3f lbn = model->getLBN();
	Vec3f rtf = model->getRTF();

	Vec4f Plbn = Vec4f(lbn[0], lbn[1], lbn[2], 1.0);
	Vec4f Prbn = Vec4f(rtf[0], lbn[1], lbn[2], 1.0);
	Vec4f Pltn = Vec4f(lbn[0], rtf[1], lbn[2], 1.0);
	Vec4f Plbf = Vec4f(lbn[0], lbn[1], rtf[2], 1.0);
	Vec4f Pltf = Vec4f(lbn[0], rtf[1], rtf[2], 1.0);
	Vec4f Prtn = Vec4f(rtf[0], rtf[1], lbn[2], 1.0);
	Vec4f Prbf = Vec4f(rtf[0], lbn[1], rtf[2], 1.0);
	Vec4f Prtf = Vec4f(rtf[0], rtf[1], rtf[2], 1.0);

	vector<Vec4f> points3D;
	points3D.push_back(Plbn);
	points3D.push_back(Prbn);
	points3D.push_back(Pltn);
	points3D.push_back(Plbf);
	points3D.push_back(Pltf);
	points3D.push_back(Prtn);
	points3D.push_back(Prbf);
	points3D.push_back(Prtf);

	Matx44f normalization = model->getNormalization();

	Point2f lt(FLT_MAX, FLT_MAX);
	Point2f rb(-FLT_MAX, -FLT_MAX);

	for (int i = 0; i < points3D.size(); i++) {
		Vec4f p = calibrationMatrices[currentLevel] * pose * normalization * points3D[i];

		if (p[2] == 0)
			continue;

		Point2f p2d = Point2f(p[0] / p[2], p[1] / p[2]);
		projections.push_back(p2d);

		if (p2d.x < lt.x) lt.x = p2d.x;
		if (p2d.x > rb.x) rb.x = p2d.x;
		if (p2d.y < lt.y) lt.y = p2d.y;
		if (p2d.y > rb.y) rb.y = p2d.y;
	}

	boundingRect.x = lt.x;
	boundingRect.y = lt.y;
	boundingRect.width = rb.x - lt.x;
	boundingRect.height = rb.y - lt.y;
}

void View::ProjectBoundingBox(Model* model, std::vector<cv::Point2f>& projections, cv::Rect& boundingRect) {
	Vec3f lbn = model->getLBN();
	Vec3f rtf = model->getRTF();

	Vec4f Plbn = Vec4f(lbn[0], lbn[1], lbn[2], 1.0);
	Vec4f Prbn = Vec4f(rtf[0], lbn[1], lbn[2], 1.0);
	Vec4f Pltn = Vec4f(lbn[0], rtf[1], lbn[2], 1.0);
	Vec4f Plbf = Vec4f(lbn[0], lbn[1], rtf[2], 1.0);
	Vec4f Pltf = Vec4f(lbn[0], rtf[1], rtf[2], 1.0);
	Vec4f Prtn = Vec4f(rtf[0], rtf[1], lbn[2], 1.0);
	Vec4f Prbf = Vec4f(rtf[0], lbn[1], rtf[2], 1.0);
	Vec4f Prtf = Vec4f(rtf[0], rtf[1], rtf[2], 1.0);

	vector<Vec4f> points3D;
	points3D.push_back(Plbn);
	points3D.push_back(Prbn);
	points3D.push_back(Pltn);
	points3D.push_back(Plbf);
	points3D.push_back(Pltf);
	points3D.push_back(Prtn);
	points3D.push_back(Prbf);
	points3D.push_back(Prtf);

	Matx44f pose = model->getPose();
	Matx44f normalization = model->getNormalization();

	Point2f lt(FLT_MAX, FLT_MAX);
	Point2f rb(-FLT_MAX, -FLT_MAX);

	for (int i = 0; i < points3D.size(); i++) {
		Vec4f p = calibrationMatrices[currentLevel] * pose * normalization * points3D[i];

		if (p[2] == 0)
			continue;

		Point2f p2d = Point2f(p[0] / p[2], p[1] / p[2]);
		projections.push_back(p2d);

		if (p2d.x < lt.x) lt.x = p2d.x;
		if (p2d.x > rb.x) rb.x = p2d.x;
		if (p2d.y < lt.y) lt.y = p2d.y;
		if (p2d.y > rb.y) rb.y = p2d.y;
	}

	boundingRect.x = lt.x;
	boundingRect.y = lt.y;
	boundingRect.width = rb.x - lt.x;
	boundingRect.height = rb.y - lt.y;
}

Mat View::DownloadFrame(View::FrameType type) {
	Mat res;
	switch (type) {
	case MASK:
		res = Mat(height, width, CV_8UC1);
		glReadPixels(0, 0, res.cols, res.rows, GL_RED, GL_UNSIGNED_BYTE, res.data);
		break;
	case RGB:
		res = Mat(height, width, CV_8UC3);
		glReadPixels(0, 0, res.cols, res.rows, GL_RGB, GL_UNSIGNED_BYTE, res.data);
		break;
	case RGB_32F:
		res = Mat(height, width, CV_32FC3);
		glReadPixels(0, 0, res.cols, res.rows, GL_RGB, GL_FLOAT, res.data);
		break;
	case DEPTH:
		res = Mat(height, width, CV_32FC1);
		glReadPixels(0, 0, res.cols, res.rows, GL_DEPTH_COMPONENT, GL_FLOAT, res.data);
		break;
	default:
		res = Mat::zeros(height, width, CV_8UC1);
		break;
	}
	return res;
}

void View::ProjectPoints(const std::vector<cv::Point3f>& pts3d, const cv::Matx44f& pose, std::vector<cv::Point2f>& pts) {
	pts.clear();

	for (int i = 0; i < pts3d.size(); i++) {
		Vec4f p = calibrationMatrices[currentLevel] * pose * cv::Vec4f(pts3d[i].x, pts3d[i].y, pts3d[i].z, 1.0f);

		//if (p[2] == 0)
		//	continue;

		//Point2f p2d = Point2f(p[0] / p[2], p[1] / p[2]);
		float x = p[0] / p[2];
		float y = p[1] / p[2];

		if (x >= 0 && x < width && y >= 0 && y < height) {
			pts.push_back(cv::Point2f(x, y));
		}
	}
}

void View::ProjectPoints(const std::vector<cv::Point3f>& pts3d, const cv::Matx44f& pose, std::vector<cv::Point>& pts) {
	pts.clear();

	for (int i = 0; i < pts3d.size(); i++) {
		Vec4f p = calibrationMatrices[currentLevel] * pose * cv::Vec4f(pts3d[i].x, pts3d[i].y, pts3d[i].z, 1.0f);

		//if (p[2] == 0)
		//	continue;

		//Point2f p2d = Point2f(p[0] / p[2], p[1] / p[2]);
		float x = p[0] / p[2];
		float y = p[1] / p[2];

		if (x >= 0 && x < width && y >= 0 && y < height) {
			pts.push_back(cv::Point(x, y));
		}
	}
}

void View::BackProjectPoints(std::vector<cv::Point>& pts, const cv::Mat& depth_map, const cv::Matx44f& pose, std::vector<cv::Point3f>& pts3d) {
	float* depthData = (float*)depth_map.ptr<float>();
	cv::Matx44f k44 = GetCalibrationMatrix();
	const cv::Matx33f& K = GetCalibrationMatrix().get_minor<3, 3>(0, 0);
	float* K_invData = K.inv().val;
	float* pdata = pose.inv().val;

	pts3d.resize(pts.size());

	for (int i = 0; i < pts3d.size(); ++i) {
		int zidx = pts[i].y * depth_map.cols + pts[i].x;
		float depth = 1.0f - depthData[zidx];

		float D = 2.0f * zn * zf / (zf + zn - (2.0f * depth - 1.0) * (zf - zn));

		float X = D * (K_invData[0] * pts[i].x + K_invData[2]);
		float Y = D * (K_invData[4] * pts[i].y + K_invData[5]);
		float Z = D;

		pts3d[i] = cv::Point3f(
			pdata[0] * X + pdata[1] * Y + pdata[2] * Z + pdata[3],
			pdata[4] * X + pdata[5] * Y + pdata[6] * Z + pdata[7],
			pdata[8] * X + pdata[9] * Y + pdata[10] * Z + pdata[11]);
	}
}