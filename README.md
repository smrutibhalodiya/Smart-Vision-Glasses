# Smart Vision Glasses

This project aims to develop a pair of smart vision glasses for visually impaired individuals, providing assistance in both reading and walking. The system utilizes advanced computer vision and deep learning techniques to detect and recognize objects, faces, currency, and text, offering audio feedback to the user.

**Overview**

The Smart Vision Glasses project is designed to help visually impaired people by offering two distinct modes: Reading Mode and Walking Mode. In Reading Mode, the glasses can detect and recognize currency, faces, and text, converting them into speech. In Walking Mode, the glasses assist with safe navigation by detecting pedestrians and providing GPS-based route guidance. The project is currently being optimized for processing on the NVIDIA Jetson Nano, with all outputs delivered via audio.

**Features**

**Reading Mode**

1) Currency Detection: The system identifies different denominations of currency using Haarcascade and CNN models.
2) Face Recognition: Recognizes familiar faces and provides auditory feedback.
3) Text-to-Speech: Converts detected text into speech. Currently, the system can recognize and speak individual words.

**Walking Mode**

1) Pedestrian Detection: Identifies pedestrians in the path to avoid collisions.
2) GPS Navigation: Provides real-time GPS-based navigation for finding the shortest route.

**Multimodal Integration**

We are currently working on a multimodal model that integrates all the functionalities of the Reading and Walking Modes. This model will preprocess and optimize these tasks on the NVIDIA Jetson Nano, ensuring efficient and real-time performance. All outputs will be provided through an audio interface.

**Future Plans**

1) **Full Book Reading:** Enhance text-to-speech capabilities to enable reading of entire books.
2) **Obstacle-Free GPS Navigation:** Integrate advanced obstacle detection for a seamless navigation experience.
3) **Complete Multimodal Integration:** Finalize the multimodal system for full integration and real-time processing on the Jetson Nano, with all outputs in audio format.
