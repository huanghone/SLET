# An Occlusion-aware Edge-Based Method for Monocular 3D Object Tracking using Edge Confidence

We propose an edge-based method for 6DOF pose tracking of rigid objects using a monocular RGB camera. One of the critical problem for edge-based methods is to search the object contour points in the image corresponding to the known 3D model points. However, previous methods often produce false object contour points in case of cluttered backgrounds and partial occlusions. In this paper, we propose a novel edge-based 3D objects tracking method to tackle this problem. To search the object contour points, foreground and background clutter points are first filtered out using edge color cue, then object contour points are searched by maximizing their edge confidence which combines edge color and distance cues. Furthermore, the edge confidence is integrated into the edge-based energy function to reduce the influence of false contour points caused by cluttered backgrounds and partial occlusions. We also extend our method to multi-object tracking which can handle mutual occlusions. We compare our method with the recent state-of-art methods on challenging public datasets. Experiments demonstrate that our method improves robustness and accuracy against cluttered backgrounds and partial occlusions.

### Preview Video

[![PG'20 supplementary video.](https://img.youtube.com/vi/-LoPCaPWs70/0.jpg)](https://www.youtube.com/watch?v=-LoPCaPWs70)


### Related Papers

* **An Occlusion-aware Edge-Based Method for Monocular 3D Object Tracking using Edge Confidence**
*Hong Huang, Fan Zhong, Xueying Qin*, PG '20

If you use SLET in your research work, please cite:

	@article{Huang2020,
		AUTHOR  = {Hong Huang and Fan Zhong and Yuqing Sun and Xueying Qin},
		TITLE   = {An occlusion‐aware edge‐based method for monocular 3D object tracking using edge confidence},
		JOURNAL = "Computer Graphics Forum",
		VOLUME  = {39}, 
		NUMBER  = {7},
		YEAR    = {2020},	
		PAGES   = {399-409},
	}

# Dependencies

SLET depends on (recent versions of) the following software libraries:

* Assimp
* OpenCV
* OpenGL
* Qt

**It must be run from the root directory (that contains the *src* folder) otherwise the relative paths to the model and the shaders will be wrong.**

The code was developed and tested under Windows 10. It should, however, also run on IOS and Linux systems with (probably) a few minor changes required. Nothing is plattform specific by design.


# How To Use

The general usage of the algorithm is demonstrated in a small example command line application provided in `run_on_video.cc`. Here the pose of a single 3D model is refined with respect to a given example image. The extension to actual pose tracking and using multiple objects should be straight foward based on this example. Simply replace the example image with the live feed from a camera or a video and add your own 3D models instead.

For the best performance when using your own 3D models, please **ensure that each 3D model consists of a maximum of around 4000 - 7000 vertices and is equally sampled across the visible surface**. This can be enforced by using a 3D mesh manipulation software such as MeshLab (http://www.meshlab.net/) or OpenFlipper (https://www.openflipper.org/).


# Dataset

To test the algorithm you can for example use the corresponding dataset available for download at: https://github.com/huanghone/RBOTE


# License

SLET is licensed under the GNU General Public License Version 3 (GPLv3), see http://www.gnu.org/licenses/gpl.html.
