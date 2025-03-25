# CS5330 - Lab 4: 3D Image Composer

## Overview
This project is a **3D Image Composer** that allows users to **insert a segmented person** into a **stereoscopic background** and generate a **red-cyan anaglyph 3D image**.  
The application is built using **Python, OpenCV, PyTorch, and Gradio**.

## Features
- **Semantic Segmentation**: Uses **DeepLabV3-ResNet101** to segment a person from an input image.
- **Stereo Backgrounds**: Supports **preset stereo backgrounds**.
- **Depth Simulation**: Inserts the segmented person at different depth levels (**Close, Medium, Far**) using **disparity shifts**.
- **3D Anaglyph Generation**: Creates a **red-cyan anaglyph image** for viewing with 3D glasses.
- **User-Friendly UI**: Implemented using **Gradio** for easy interaction.
