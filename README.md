# Deep Reinforcement One-Shot Learning (DeROL) Classification Framework

This repository contains the DeROL framework code, as presented in "Deep Reinforcement One-Shot Learning for Artificially Intelligent Classification Systems" [link will be added here]

The paper has been uploaded to arXiv (identifier 1808.01527) and is accessible from: http://arxiv.org/abs/1808.01527



## Special Notes
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
* Any code/data piece supplementary to this repository **must** be used in accordance to its own license and terms
* [`datasets`](datasets) has the instructions for obtaining the datasets
* If this code was used in your research please cite our paper: [BibTeX](https://github.com/antonpuz/DeROL#please-cite-our-paper)

## Requirements
* We ran the code on Ubuntu 16.04.4 LTS but it should be easily used in MacOS, using Windows would require fixing all the paths in "train_derol.py"
* We used Python 2.7.12 with the following packages: tensorflow, numpy, scipy, sklearn

## Intoduction
In recent years there has been a sharp rise in networking applications, in which significant events need to be classified but only a few training instances are available. These are known as cases of one-shot learning. Examples include analyzing network traffic under zero-day attacks, and computer vision tasks by sensor networks deployed in the field. To handle this challenging task, organizations often use human analysts to classify events under high uncertainty. Existing algorithms use a threshold-based mechanism to decide whether to classify an object automatically or send it to an analyst for deeper inspection. However, this approach leads to a significant waste of resources since it does not take the practical temporal constraints of system resources into account. Our contribution is threefold. First, we develop a novel Deep Reinforcement One-shot Learning (DeROL) framework to address this challenge. The basic idea of the DeROL algorithm is to train a deep-Q network to obtain a policy which is oblivious to the unseen classes in the testing data. Then, in real-time, DeROL maps the current state of the one- shot learning process to operational actions based on the trained deep-Q network, to maximize the objective function. Second, we develop the first open-source software for practical artificially intelligent one-shot classification systems with limited resources for the benefit of researchers and developers in related fields. Third, we present an extensive experimental study using the OMNIGLOT dataset for computer vision tasks and the UNSW- NB15 dataset for intrusion detection tasks that demonstrates the versatility and efficiency of the DeROL framework.

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
    @ARTICLE{__,
        author  = {Puzanov Anton and Cohen Kobi},
        journal = {submitted to IEEE Journal on Selected Areas in Communications, preliminary version is available at arXiv},
        title   = {Deep Reinforcement One-Shot Learning for Artificially Intelligent Classification Systems},
        year 	= {2018},
        volume 	= {__},
        number 	= {__},
        pages 	= {__-__},
        doi 	= {__},
        ISSN 	= {__},
        month 	= {__},
    }