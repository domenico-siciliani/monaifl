<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Thanks again! Now go create something AMAZING! :D
***
***
***
*** To avoid retyping too much info. Do a search and replace for the following:
*** github_username, repo_name, twitter_handle, email, project_title, project_description
-->

<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
<!--[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]
-->

<!-- PROJECT LOGO -->
<br />
<!--<p align="center">
  <a href="https://github.com/domenicosiciliani-digica/monaifl">
    <img src="images/logo.png" alt="Logo" width="120" height="80">
  </a>
-->
  <h3 align="center">MONAI-FL</h3>

  <p align="center">
    This repository contains the basic federated learning example of monai code considering substra's opener and algo classes.
    <!--<br />
    <a href="https://github.com/github_username/repo_name"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/github_username/repo_name">View Demo</a>
    ·
    <a href="https://github.com/github_username/repo_name/issues">Report Bug</a>
    ·
    <a href="https://github.com/github_username/repo_name/issues">Request Feature</a>
  </p>
    -->
</p>



<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary><h2 style="display: inline-block">Table of Contents</h2></summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## 1. About The Project
<p align="left">
MONAI-FL is planned to be a tool-agnostic federated learning platform for medical imaging applications. We envision the below-mentioned architecture for our code-base, however, please feel free to modify the code-base considering your own application requirements.
</p>
<p align="center">
<img src="images/monaifl.png" alt="MONAI-FL" width="1280" height="720" align="center">
</p>
<!--
Here's a blank template to get started:
**To avoid retyping too much info. Do a search and replace with your text editor for the following:**
`github_username`, `repo_name`, `twitter_handle`, `email`, `project_title`, `project_description`
-->

### Built With

* [Python 3.8](https://www.python.org/downloads/release/python-380/)
* [Google gRPC](https://grpc.io/)
* [Project MONAI](https://monai.io/)
* [Pytorch](https://pytorch.org/get-started/locally/)
* [Ubuntu](https://ubuntu.com/)



<!-- GETTING STARTED -->
## 2. Getting Started

To get a local copy up and running follow these simple steps.

### Prerequisites

Following platforms are neccessary to run this project.
* Linux Installation (preferably Ubuntu distributions)

* install visual studio code
  ```sh
  https://code.visualstudio.com/docs/setup/linux
  ```

### Installation

1. Install Python 3.8
   ```sh
   pip install python==3.8
   ```
2. Clone the monaifl repository in your project's directory
   ```sh
   git clone https://github.com/domenicosiciliani-digica/monaifl.git
   ```

3. Go to the monaifl directory
   ```sh
   cd monaifl
   ```

2. Clone the MONAI repository
   ```sh
   git clone https://github.com/Project-MONAI/MONAI.git
   ```
3. Go back to the monaifl directory
   ```sh
   cd .. 
   ```

3. Run the command to create the virtuall enviroment and install the dependencies specified in the requirements.txt tile
   ```sh
   virtualenv venv && venv/bin/pip install -r requirements.txt -f https://download.pytorch.org/whl/cu113/torch_stable.html
   ```
4. Activate the virtuall enviroment by running:
   ```sh
   source venv/bin/activate
   ```

<!-- CONTRIBUTING -->
## 3. Contributing

1. Fork the Project
2. Create your New Branch (`git checkout -b "branch name"`)
3. Find a way to fix / make the code run better
4. Commit your Changes (`git commit -m 'comment describing the change you've made`)
5. Push to the Branch (`git push origin "branch name"`)
6. Open a Pull Request

<!-- LICENSE -->
## 4. License

Distributed under the MIT License: https://opensource.org/licenses/MIT

<!-- CONTACT -->
## 5. Contact

Project Link: [https://github.com/domenicosiciliani-digica/monaifl](https://github.com/domenicosiciliani-digica/monaifl)


<!-- ACKNOWLEDGEMENTS -->
## 6. Acknowledgements

* [AMIGO](https://github.com/AmigoLab)
* [AnswerDigital](https://answerdigital.com/)
* [AI Center for Value-Based Healthcare](https://www.aicentre.co.uk/)

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
<!--[contributors-shield]: https://img.shields.io/github/contributors/github_username/repo.svg?style=for-the-badge
[contributors-url]: https://github.com/github_username/repo_name/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/github_username/repo.svg?style=for-the-badge
[forks-url]: https://github.com/github_username/repo_name/network/members
[stars-shield]: https://img.shields.io/github/stars/github_username/repo.svg?style=for-the-badge
[stars-url]: https://github.com/github_username/repo_name/stargazers
[issues-shield]: https://img.shields.io/github/issues/github_username/repo.svg?style=for-the-badge
[issues-url]: https://github.com/github_username/repo_name/issues
[license-shield]: https://img.shields.io/github/license/github_username/repo.svg?style=for-the-badge
[license-url]: https://github.com/github_username/repo_name/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/github_username
-->
