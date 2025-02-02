# Use a stable Ubuntu version
FROM ubuntu:22.04

# Set environment variables
ENV DISPLAY=:1
ENV DEBIAN_FRONTEND=noninteractive
ENV VNC_PASSWORD=password

# Update packages and install necessary dependencies
RUN apt-get update && apt-get upgrade -y && apt-get install -y \
    software-properties-common build-essential cmake git \
    python3 python3-pip python3-venv \
    python3-tk python3-opencv \
    ttf-wqy-zenhei \
    libsm6 libxrender1 libxext6 \
    x11vnc xvfb fluxbox \
    tigervnc-standalone-server tigervnc-tools \
    libzbar0 zbar-tools \
    novnc websockify \
    && apt-get clean

# Create a Python virtual environment
RUN python3 -m venv /opt/venv

# Install Python packages in the virtual environment
RUN /opt/venv/bin/pip install --no-cache-dir \
    numpy \
    matplotlib \
    opencv-python-headless \
    tensorflow \
    flask \
    flask-cors \
    barcode \
    python-barcode \
    pillow \
    pyzbar \
    heapq_max

# Set the PATH for the virtual environment
ENV PATH="/opt/venv/bin:$PATH"

# Configure VNC
RUN mkdir -p ~/.vnc && \
    x11vnc -storepasswd ${VNC_PASSWORD} ~/.vnc/passwd && \
    chmod 600 ~/.vnc/passwd

RUN echo "fluxbox &" > ~/.vnc/xstartup && chmod +x ~/.vnc/xstartup

# Set working directory and copy application files
WORKDIR /app
COPY . .

# Expose VNC, Flask, and noVNC ports
EXPOSE 5901 5000 6080

# Start services: Xvfb, VNC, noVNC, and Flask
CMD [ "sh", "-c", "Xvfb :1 -screen 0 1280x800x24 & x11vnc -display :1 -rfbport 5901 -forever -passwd ${VNC_PASSWORD} & websockify --web /usr/share/novnc/ 6080 localhost:5901 & python3 main.py" ]