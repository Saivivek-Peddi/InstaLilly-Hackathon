#!/bin/bash
# Sync local files to VM using rsync
#
# Usage:
#   ./scripts/sync_to_vm.sh              # Sync entire project
#   ./scripts/sync_to_vm.sh pedestrian   # Sync only pedestrian_detector
#   ./scripts/sync_to_vm.sh models       # Sync only models folder

VM_USER="hackathon"
VM_HOST="34.123.38.202"
VM_PATH="/home/hackathon/InstaLilly-Hackathon"

LOCAL_PATH="$(cd "$(dirname "$0")/.." && pwd)"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 Syncing to VM: ${VM_USER}@${VM_HOST}${NC}"
echo "   Local:  $LOCAL_PATH"
echo "   Remote: $VM_PATH"
echo ""

# Common rsync options
RSYNC_OPTS="-avz --progress --exclude='.venv' --exclude='__pycache__' --exclude='.git' --exclude='*.pyc' --exclude='.DS_Store' --exclude='data/processed' --exclude='checkpoints'"

case "$1" in
    pedestrian)
        echo -e "${YELLOW}📁 Syncing pedestrian_detector only...${NC}"
        rsync $RSYNC_OPTS \
            "$LOCAL_PATH/models/pedestrian_detector/" \
            "${VM_USER}@${VM_HOST}:${VM_PATH}/models/pedestrian_detector/"
        ;;
    drowsiness)
        echo -e "${YELLOW}📁 Syncing drowsiness_detector only...${NC}"
        rsync $RSYNC_OPTS \
            "$LOCAL_PATH/models/drowsiness_detector/" \
            "${VM_USER}@${VM_HOST}:${VM_PATH}/models/drowsiness_detector/"
        ;;
    distraction)
        echo -e "${YELLOW}📁 Syncing distraction_detector only...${NC}"
        rsync $RSYNC_OPTS \
            "$LOCAL_PATH/models/distraction_detector/" \
            "${VM_USER}@${VM_HOST}:${VM_PATH}/models/distraction_detector/"
        ;;
    voice)
        echo -e "${YELLOW}📁 Syncing voice_assistant only...${NC}"
        rsync $RSYNC_OPTS \
            "$LOCAL_PATH/models/voice_assistant/" \
            "${VM_USER}@${VM_HOST}:${VM_PATH}/models/voice_assistant/"
        ;;
    models)
        echo -e "${YELLOW}📁 Syncing all models...${NC}"
        rsync $RSYNC_OPTS \
            "$LOCAL_PATH/models/" \
            "${VM_USER}@${VM_HOST}:${VM_PATH}/models/"
        ;;
    scripts)
        echo -e "${YELLOW}📁 Syncing scripts...${NC}"
        rsync $RSYNC_OPTS \
            "$LOCAL_PATH/scripts/" \
            "${VM_USER}@${VM_HOST}:${VM_PATH}/scripts/"
        ;;
    *)
        echo -e "${YELLOW}📁 Syncing entire project...${NC}"
        rsync $RSYNC_OPTS \
            "$LOCAL_PATH/" \
            "${VM_USER}@${VM_HOST}:${VM_PATH}/"
        ;;
esac

echo ""
echo -e "${GREEN}✅ Sync complete!${NC}"
echo ""
echo "To run on VM:"
echo "  ssh ${VM_USER}@${VM_HOST}"
echo "  cd ${VM_PATH}"
echo "  python models/pedestrian_detector/download_samples.py --full --kaggle"
