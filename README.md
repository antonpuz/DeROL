# Deep Reinforcement One-Shot Learning (DeROL) Classification Framework

This repository contains the DeROL framework code, as presented in "Deep Reinforcement One-Shot Learning for Artificially Intelligent Classification Systems" [link will be added here]

## Special Notes
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
* Any code/data piece supplementary to this repository **must** be used in accordance to its own license
* ['datasets'] has the instructions for obtaining the datasets
* If this code was used in your research please cite our paper (BibTeX added below)

## Requirements
* We ran the code on Ubuntu 16.04.4 LTS but it should be easily used in MacOS, using Windows would require fixing all the paths in "train_derol.py"
* We used Python 2.7.12 with the following packages: tensorflow, numpy, scipy, sklearn

## Intoduction
Recent years have witnessed various analytical applications, in which classifying significant events is required, but only a few (or even none) training instances are avail- able, known as one-shot learning, prominent example include network applications where network traffic holds previously un-seen zero day cyber attacks, or computer vision tasks where labeled samples are extremely hard to get such as x-ray scans. To handle this challenging task, organizations often use human analysts to assist in classifying events under high uncertainty. Existing algorithms use a threshold-based mechanism to decide whether to classify an object automatically or send it to an analyst for deeper inspection. However, this approach leads to a significant waste of resources since it does not take into account practical time-varying system resources (e.g., analysts time, task load, etc.). Our contribution is threefold. First, we develop a novel Deep Reinforcement One-Shot Learning (DeROL) framework to address this challenge, and published it as open-source project for future use. The DeROL framework improves the one-shot learning classification performance by learning online from human actions, with the goal of minimizing the total system cost. Second, we develop a first open-source software for artificially intelligent classification systems for the benefit of the networking community. Third, we present an extensive experimental study using OMNIGLOT dataset for computer vision tasks and UNSW- NB15 dataset for intrusion detection tasks that demonstrates the versatility and efficiency of the DeROL framework.

## Modules
* **rl_agents**: Has implementation of the policies, the policy is the conductor and the heart of our framework, we used Deep Reinforcement Learning policy,
with an Long Short Term Memory component and 1 hidden layer.
* **classifiers**: The classifier outputs similarity measure to all the known classes, almost every dataset would require a either new or adapted classifier.
The classifier must support one-shot behavior, we used samples bank with a distance metric to predict the similarity to all the classes. We implemented classifiers for two different datasets:
For OMNIGLOT dataset: we used [“A Modified Hausdorff Distance for Object Matching”](http://www.cse.msu.edu/prip/Files/DubuissonJain.pdf) as distance metric
For UNSW0NB15: we used [“Euclidean Distance”](https://en.wikipedia.org/wiki/Euclidean_distance) as distance metric
* **analyst_manager**: is used to manage the analyst's work cycle
* **delay**: holds random delay creators to be used in the framework
* **sample_generators**: Has implementation of generators, which are responsible for reading data saved on disk, and create ordered batches from it.
We implemented two types of generator, one for every dataset.
OMNIGLOTGenerator: supports reading image samples from OMNIGLOT dataset, returns either raw pixel values or image names
UNSWGenerator: supports reading "csv" files, and normalize them according to pre-calculated mean and standard deviation values.
* **sample_handlers**: has the classes which support extra sample holding classes,
DelayClassification: holds samples marked to be delayed by the policy
ExperimentLogger: holds {sample, action, reward, future_reward} tuples, used later for training.

## Please Cite Our Paper


