#include "object3d.hh"
#include "tclc_histograms.hh"
#include "template_view.hh"

using namespace std;
using namespace cv;

bool sortDistance(std::pair<float, int> a, std::pair<float, int> b)
{
    return a.first < b.first;
}

void Object3D::Init(float qualityThreshold, std::vector<float> &templateDistances) {
    this->trackingLost = false;
    this->qualityThreshold = qualityThreshold;
    this->templateDistances = templateDistances;
    this->numDistances = (int)templateDistances.size();
    this->tclcHistograms = NULL;
    
    // icosahedron geometry for generating the base templates
    baseIcosahedron.push_back(Vec3f(0, 1, 1.61803));
    baseIcosahedron.push_back(Vec3f(1, 1.61803, 0));
    baseIcosahedron.push_back(Vec3f(-1, 1.61803, 0));
    baseIcosahedron.push_back(Vec3f(0, 1, -1.61803));
    baseIcosahedron.push_back(Vec3f(1.61803, 0, -1));
    baseIcosahedron.push_back(Vec3f(0, -1, -1.61803));
    baseIcosahedron.push_back(Vec3f(-1.61803, 0, -1));
    baseIcosahedron.push_back(Vec3f(-1, -1.61803, 0));
    baseIcosahedron.push_back(Vec3f(-1.61803, 0, 1));
    baseIcosahedron.push_back(Vec3f(0, -1, 1.61803));
    baseIcosahedron.push_back(Vec3f(1.61803, 0, 1));
    baseIcosahedron.push_back(Vec3f(1, -1.61803, 0));
    
    // sub-divided icosahedron geometry for generating the neighboring templates
    subdivIcosahedron.push_back(Vec3f(0, 1.90811, 0));
    subdivIcosahedron.push_back(Vec3f(-0.589637, 1.54369, 0.954053));
    subdivIcosahedron.push_back(Vec3f(0.589637, 1.54369, 0.954053));
    subdivIcosahedron.push_back(Vec3f(0.589637, 1.54369, -0.954053));
    subdivIcosahedron.push_back(Vec3f(-0.589637, 1.54369, -0.954053));
    subdivIcosahedron.push_back(Vec3f(0, 0, -1.90811));
    subdivIcosahedron.push_back(Vec3f(0.954053, 0.589637, -1.54369));
    subdivIcosahedron.push_back(Vec3f(0.954053, -0.589637, -1.54369));
    subdivIcosahedron.push_back(Vec3f(-0.954053, -0.589637, -1.54369));
    subdivIcosahedron.push_back(Vec3f(-0.954053, 0.589637, -1.54369));
    subdivIcosahedron.push_back(Vec3f(-1.90811, 0, 0));
    subdivIcosahedron.push_back(Vec3f(-1.54369, -0.954053, -0.589637));
    subdivIcosahedron.push_back(Vec3f(-1.54369, -0.954053, 0.589637));
    subdivIcosahedron.push_back(Vec3f(-1.54369, 0.954053, 0.589637));
    subdivIcosahedron.push_back(Vec3f(-1.54369, 0.954053, -0.589637));
    subdivIcosahedron.push_back(Vec3f(0, 0, 1.90811));
    subdivIcosahedron.push_back(Vec3f(-0.954053, 0.589637, 1.54369));
    subdivIcosahedron.push_back(Vec3f(-0.954053, -0.589637, 1.54369));
    subdivIcosahedron.push_back(Vec3f(0.954053, -0.589637, 1.54369));
    subdivIcosahedron.push_back(Vec3f(0.954053, 0.589637, 1.54369));
    subdivIcosahedron.push_back(Vec3f(1.90811, 0, 0));
    subdivIcosahedron.push_back(Vec3f(1.54369, -0.954053, 0.589637));
    subdivIcosahedron.push_back(Vec3f(1.54369, -0.954053, -0.589637));
    subdivIcosahedron.push_back(Vec3f(1.54369, 0.954053, -0.589637));
    subdivIcosahedron.push_back(Vec3f(1.54369, 0.954053, 0.589637));
    subdivIcosahedron.push_back(Vec3f(0, -1.90811, 0));
    subdivIcosahedron.push_back(Vec3f(-0.589637, -1.54369, -0.954053));
    subdivIcosahedron.push_back(Vec3f(0.589637, -1.54369, -0.954053));
    subdivIcosahedron.push_back(Vec3f(0.589637, -1.54369, 0.954053));
    subdivIcosahedron.push_back(Vec3f(-0.589637, -1.54369, 0.954053));
    subdivIcosahedron.push_back(Vec3f(-1.00074, 1.61923, 0));
    subdivIcosahedron.push_back(Vec3f(0, 1.00074, 1.61923));
    subdivIcosahedron.push_back(Vec3f(1.00074, 1.61923, 0));
    subdivIcosahedron.push_back(Vec3f(0, 1.00074, -1.61923));
    subdivIcosahedron.push_back(Vec3f(1.61923, 0, -1.00074));
    subdivIcosahedron.push_back(Vec3f(0, -1.00074, -1.61923));
    subdivIcosahedron.push_back(Vec3f(-1.61923, 0, -1.00074));
    subdivIcosahedron.push_back(Vec3f(-1.00074, -1.61923, 0));
    subdivIcosahedron.push_back(Vec3f(-1.61923, 0, 1.00074));
    subdivIcosahedron.push_back(Vec3f(0, -1.00074, 1.61923));
    subdivIcosahedron.push_back(Vec3f(1.61923, 0, 1.00074));
    subdivIcosahedron.push_back(Vec3f(1.00074, -1.61923, 0));
}

Object3D::Object3D(const std::string objFilename, const cv::Matx44f& Ti, float scale, float qualityThreshold, std::vector<float> &templateDistances) 
	: Model(objFilename, Ti, scale)
{
	Init(qualityThreshold, templateDistances);
}

Object3D::Object3D(const string objFilename, float tx, float ty, float tz, float alpha, float beta, float gamma, float scale, float qualityThreshold,  vector<float> &templateDistances) 
	: Model(objFilename, tx, ty, tz, alpha, beta, gamma, scale)
{
	Init(qualityThreshold, templateDistances);
}


Object3D::~Object3D()
{
    delete tclcHistograms;
    
    for(int i = 0; i < baseTemplates.size(); i++)
    {
        delete baseTemplates[i];
    }
    baseTemplates.clear();
    
    for(int i = 0; i < neighboringTemplates.size(); i++)
    {
        delete neighboringTemplates[i];
    }
    neighboringTemplates.clear();
}


bool Object3D::isTrackingLost()
{
    return trackingLost;
}

void Object3D::setTrackingLost(bool val)
{
    trackingLost = val;
}

float Object3D::getQualityThreshold()
{
    return qualityThreshold;
}


TCLCHistograms *Object3D::getTCLCHistograms()
{
    return tclcHistograms;
}

void Object3D::SetTCLCHistograms(TCLCHistograms* histograms) {
	tclcHistograms = histograms;
}

void Object3D::generateTemplates()
{
    int numLevels = 4;
    
    int numBaseRotations = 4;
    
    // create all base templates
    for(int i = 0; i < baseIcosahedron.size(); i++)
    {
        Vec3f v = baseIcosahedron[i];
        
        float r = norm(v);
        float alpha = acos(v[1]/r)*180.0f/float(CV_PI) - 90.0f;
        float beta = atan2(v[0], v[2])*180.0f/float(CV_PI);
        
        for(int gamma = 0; gamma < 360; gamma += 90)
        {
            for(int d = 0; d < numDistances; d++)
            {
                baseTemplates.push_back(new TemplateView(this, alpha, beta, gamma, templateDistances[d], numLevels, true));
            }
        }
    }
    
    int gamma2Precision = 30;
    
    // create all neighboring templates
    for(int i = 0; i < subdivIcosahedron.size(); i++)
    {
        Vec3f v = subdivIcosahedron[i];
        
        float r = norm(v);
        float alpha = acos(v[1]/r)*180.0f/float(CV_PI) - 90.0f;
        float beta = atan2(v[0], v[2])*180.0f/float(CV_PI);
        
        for(int gamma = 0; gamma < 360; gamma += gamma2Precision)
        {
            for(int d = 0; d < numDistances; d++)
            {
                neighboringTemplates.push_back(new TemplateView(this, alpha, beta, gamma, templateDistances[d], numLevels, true));
            }
        }
    }
    
    int gamma2Steps = 360/gamma2Precision;
    
    // associate each base template with its corresponding neighboring templates
    for(int i = 0; i < baseIcosahedron.size(); i++)
    {
        Vec3f v1 = baseIcosahedron[i];
        
        vector<pair<float, int> > distanceMap;
        
        for(int j = 0; j < subdivIcosahedron.size(); j++)
        {
            Vec3f v2 = subdivIcosahedron[j];
            
            float d = norm(v1 - v2);
            distanceMap.push_back(pair<float, int>(d, j));
        }
        
        sort(distanceMap.begin(), distanceMap.end(), sortDistance);
        
        for(int n = 0; n < 6; n++)
        {
            for(int g = 0; g < numBaseRotations; g++)
            {
                for(int d = 0; d < numDistances; d++)
                {
                    TemplateView *kv = baseTemplates[i*numBaseRotations*numDistances + numDistances*g + d];
                    float gamma1 = kv->getGamma();
                    
                    int g2 = gamma1/gamma2Precision;
                    
                    kv->addNeighborTemplate(neighboringTemplates[distanceMap[n].second*gamma2Steps*numDistances + g2*numDistances + d]);
                    
                    int g3 = (g2+gamma2Steps+1)%gamma2Steps;
                    kv->addNeighborTemplate(neighboringTemplates[distanceMap[n].second*gamma2Steps*numDistances + g3*numDistances + d]);
                    
                    int g4 = (g2+gamma2Steps-1)%gamma2Steps;
                    kv->addNeighborTemplate(neighboringTemplates[distanceMap[n].second*gamma2Steps*numDistances + g4*numDistances + d]);
                }
            }
        }
    }
    
    // reset the model to its prescribed initial pose
    Model::reset();
}


vector<TemplateView*> Object3D::getTemplateViews()
{
    return baseTemplates;
}


int Object3D::getNumDistances()
{
    return numDistances;
}


void Object3D::reset()
{
    Model::reset();
    
		if (tclcHistograms)
			tclcHistograms->clear();
    
    trackingLost = false;
}
