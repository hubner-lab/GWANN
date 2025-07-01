#!/bin/bash

extract() {
    echo "Downloading $filename ..."
    if command -v curl >/dev/null 2>&1; then
        curl -O "$url"
    elif command -v wget >/dev/null 2>&1; then
        wget "$url"
    else
        echo "Neither curl nor wget is installed. Please install one to proceed."
        exit 1
    fi

    echo "Extracting $filename ..."
    case $filename in
        *.tar.gz) tar -xzf "$filename" ;;
        *.zip) unzip "$filename" ;;
        *) echo "Unknown file format. Cannot extract."; exit 1 ;;
    esac

    echo "Done: $filename downloaded and extracted."
}

echo "Select which GENOME dependency to download:"
echo "1) Source code (tar.gz)"
echo "2) Source code (zip)"
echo "3) Precompiled binary for Linux"
echo "4) Precompiled binary for Windows"
echo "5) Precompiled binary for SunOS"
read -p "Enter your choice [1-5]: " choice

case $choice in
    1)
        url="https://csg.sph.umich.edu/liang/genome/genome-0.2.tar.gz"
        filename="genome-0.2.tar.gz"
        extract
        ;;
    2)
        url="https://csg.sph.umich.edu/liang/genome/genome-0.2.zip"
        filename="genome-0.2.zip"
        extract
        ;;
    3)
        echo "Do you want the 32-bit or 64-bit Linux binary?"
        echo "1) 32-bit"
        echo "2) 64-bit"
        read -p "Enter your choice [1-2]: " linux_choice
        case $linux_choice in
            1)
                url="https://csg.sph.umich.edu/liang/genome/genome-0.2-Linux.tar.gz"
                filename="genome-0.2-Linux.tar.gz"
                extract
                mv genome-0.2-Linux/genome-linux-32bit genome
                ;;
            2)
                url="https://csg.sph.umich.edu/liang/genome/genome-0.2-Linux.tar.gz"
                filename="genome-0.2-Linux.tar.gz"
                extract
                mv genome-0.2-Linux/genome-linux-64bit genome
                ;;
            *)
                echo "Invalid choice. Exiting."
                exit 1
                ;;
        esac
        rm genome-0.2-Linux.tar.gz
        ;;
    4)
        url="https://csg.sph.umich.edu/liang/genome/genome-0.2-Windows.zip"
        filename="genome-0.2-Windows.zip"
        extract
        mv genome-0.2-Windows/genome.exe genome
        rm genome-0.2-Windows.zip
        ;;
    5)
        url="https://csg.sph.umich.edu/liang/genome/genome-0.2-SunOS.tar.gz"
        filename="genome-0.2-SunOS.tar.gz"
        extract
        ;;
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac
