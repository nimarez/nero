#!/bin/bash
set -euo pipefail

config=/opt/booster/perception_info.yaml
backup=/opt/booster/perception_info.yaml.nero-backup
service=booster-daemon-perception.service

if [[ ! -f "$config" ]] || ! systemctl list-unit-files "$service" >/dev/null 2>&1; then
    echo "Booster K1 perception files were not found; run this on the robot." >&2
    exit 1
fi

sudo cp -pn "$config" "$backup"
if grep -q '^EnableCameraBridge:' "$config"; then
    sudo sed -i 's/^EnableCameraBridge:.*/EnableCameraBridge: true/' "$config"
else
    printf '%s\n' 'EnableCameraBridge: true' | sudo tee -a "$config" >/dev/null
fi
sudo systemctl enable "$service"

echo "Configured Booster RGB-D bridge and enabled $service for future boots."
echo "A backup remains at $backup. Reboot before running Nero."
