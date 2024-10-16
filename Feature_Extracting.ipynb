{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "machine_shape": "hm",
      "gpuType": "V28",
      "mount_file_id": "1iai7jJ3S7u1_OO_jgg8-TlnUqYSf829w",
      "authorship_tag": "ABX9TyN1H07yQ4/PE8MX77PL+1GJ"
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    },
    "accelerator": "TPU"
  },
  "cells": [
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "033Kgr3A3U74"
      },
      "outputs": [],
      "source": [
        "import cv2 as cv\n",
        "import numpy as np\n",
        "from skimage.feature import graycomatrix, graycoprops\n",
        "import os"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "def extract_color_features(image):\n",
        "\n",
        "  #rgb\n",
        "\n",
        "  color_features = cv.mean(image)[:3]\n",
        "\n",
        "  #hsv\n",
        "\n",
        "  hsv_images = cv.cvtColor(image, cv.COLOR_BGR2HSV)\n",
        "  hsv_features = cv.mean(hsv_images)[:3]\n",
        "\n",
        "  return np.concatenate([color_features, hsv_features])"
      ],
      "metadata": {
        "id": "8FhGKJ5V4C5u"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "#doku özellikleri\n",
        "\n",
        "def extract_glcm_features(image):\n",
        "  gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)\n",
        "  glcm = graycomatrix(gray, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4])\n",
        "  contrast = graycoprops(glcm, 'contrast')\n",
        "  dissimilarity = graycoprops(glcm, 'dissimilarity')\n",
        "  homogeneity = graycoprops(glcm, 'homogeneity')\n",
        "  energy = graycoprops(glcm, 'energy')\n",
        "  correlation = graycoprops(glcm, 'correlation')\n",
        "\n",
        "  return np.concatenate([contrast, dissimilarity, homogeneity, energy, correlation], axis=0)"
      ],
      "metadata": {
        "id": "O8VeVA3z7I9P"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "#kenar özellikleri\n",
        "\n",
        "def extract_edge_features(image):\n",
        "  gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)\n",
        "  edges = cv.Canny(gray, 100, 200)\n",
        "  edge_density = np.mean(edges/255.0)\n",
        "  return edge_density"
      ],
      "metadata": {
        "id": "LsXAiX3E7R_W"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "#tüm özellikleri tek fonksiyonda çağırma\n",
        "\n",
        "def extract_all_features(image):\n",
        "  color_features = extract_color_features(image)\n",
        "  glcm_features = extract_glcm_features(image)\n",
        "  edge_features = extract_edge_features(image)\n",
        "\n",
        "  color_features = np.ravel(color_features)\n",
        "  glcm_features = np.ravel(glcm_features)\n",
        "  edge_features = np.ravel(edge_features)\n",
        "\n",
        "  return np.concatenate([color_features, glcm_features, edge_features])"
      ],
      "metadata": {
        "id": "ibruG1wK8Srb"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "def process_dataset(dataset_path):\n",
        "\n",
        "  features = []\n",
        "  labels = []\n",
        "\n",
        "  for class_name in os.listdir(dataset_path):\n",
        "    class_path = os.path.join(dataset_path, class_name)\n",
        "\n",
        "    if os.path.isdir(class_path):\n",
        "      for image_name in os.listdir(class_path):\n",
        "        image_path = os.path.join(class_path,image_name)\n",
        "        image = cv.imread(image_path)\n",
        "\n",
        "        if image is not None:\n",
        "          image_features = extract_all_features(image)\n",
        "          features.append(image_features)\n",
        "          labels.append(class_name)\n",
        "  return np.array(features), np.array(labels)"
      ],
      "metadata": {
        "id": "EBWAbKcT85iC"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "dataset_path = \"/content/drive/MyDrive/Apple\"\n",
        "X, y = process_dataset(dataset_path)\n",
        "print(\"Özellik matrisinin boyutu: \", X.shape)\n",
        "print(\"Etiket vektörünün boyutu: \", y.shape)"
      ],
      "metadata": {
        "id": "w5LNKb02IPwg",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "6d8f1bb1-d198-4090-8494-852c38efff98"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Özellik matrisinin boyutu:  (148856, 27)\n",
            "Etiket vektörünün boyutu:  (148856,)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "mcNxJxrtI7fG"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}