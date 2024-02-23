
# Attempt 22: Use new dataloader in train caption
Branch from 17, and add a new data loader that does not use aligned motion. Instead, we use raw motion.

But first, we extract motion features with extract_motion_code.py

The motion code is extracted by zero-padding the remaining parts

# Attempt 22.2: Use repeated motion code
