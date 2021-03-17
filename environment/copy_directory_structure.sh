#!/bin/bash

function usage()
{
    echo "--target 'path_to_the_directory': target directory you want to copy from"
    echo "--structure_only: if you only want to copy the structure without files "
    echo "--copy_files: if you want to copy files into the directory"
    echo "--sample_videos: if you want to copy small fraction of videos, default 100 frames"
    echo "--frame '': how many frames you want to copy from the target directory"
}

while [ "$1" != "" ]; do
    case $1 in
        -h | --help)
            usage
            exit
            ;;
        --target)
            if [ -d $TARGET_DIR ]; then
                TARGET_DIR=$2
            else
                echo "$TARGET_DIR doesn't exist"
                usage
                exit
            fi
            shift
            ;;
        --copy_files)
            COPY_FILES=true
            ;;
        --sample_videos)
            SAMPLE_VIDEOS=true
            ;;
        --frame)
            NUM_FRAME=$2
            if ! [[ $NUM_FRAME =~ ^[0-9]+$ ]]
            then
                echo "Number of frames should be an integer"
                usage
                exit
            fi
            shift
            ;;
        *)
            echo "ERROR: unknown paratmeter $1"
    esac
    shift
done

if [ -z $TARGET_DIR ]; then
    echo "option --target missing."
    usage
    exit
fi

GIT_ROOT_DIR=`git rev-parse --show-toplevel`
LEAF_DIR=($(echo $TARGET_DIR | tr "/" "\n"))
LEN_LEAF_DIR=${#LEAF_DIR[@]}
LEAF_DIR=${LEAF_DIR[LEN_LEAF_DIR-1]}
TEMP_DIR=$(pwd)
DESTINATION_DIR=${GIT_ROOT_DIR}/data/${LEAF_DIR}

# Copy Directory Structure
cd $TARGET_DIR 
echo "Copy directory structure of ${TARGET_DIR} into ${DESTINATION_DIR}"
find ./ -type d -exec mkdir -p ${DESTINATION_DIR}/{} \;
cd $TEMP_DIR

# Copy Files except for Videos
if [ $COPY_FILES ]; then
    cd ${TARGET_DIR}
    readarray -d '' array < \
        <(find ./ ! -name "*.h264" \
                  ! -name "*.avi" \
                  ! -name "*.mp4" \
                  ! -name "*.mkv" \
                  ! -size +2M \
                  -type f -print0)
    LEN_FILES=${#array[@]}
    echo "Copy $LEN_FILES files from $TARGET_DIR into $DESTINATION_DIR"
    for FILE_PATH in ${array[@]}
    do
       eval `cp ${TARGET_DIR}/${FILE_PATH} ${DESTINATION_DIR}/${FILE_PATH}`
    done
    cd ${TEMP_DIR}
fi

# Copy Small Fractions of Videos
if [ $SAMPLE_VIDEOS ]; then
    cd ${TARGET_DIR}
    readarray -d '' array < <(find ./ -name "*.h264" -type f -print0; \
                              find ./ -name "*.avi" -type f -print0; \
                              find ./ -name "*.mp4" -type f -print0; \
                              find ./ -name "*.mkv" -type f -print0;
                             )
    LEN_FILES=${#array[@]}
    echo "Sample $LEN_FILES videos from $TARGET_DIR into $DESTINATION_DIR"
    if [ -z $NUM_FRAME ]; then
        NUM_FRAME=100
    fi
    for VIDEO in ${array[@]}
    do
        eval `python ${GIT_ROOT_DIR}/video_processing_utilities/sample_video.py \
            --input ${TARGET_DIR}/${VIDEO} \
            --output ${DESTINATION_DIR}/${VIDEO} \
            --frame_num ${NUM_FRAME}`
    done
    cd ${TEMP_DIR}
fi

