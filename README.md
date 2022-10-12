# CIKQA

### Step 1: Setup
1. Setup the environment with "requirements.txt"
2. Download the preprocessed data from [data](https://hkustconnect-my.sharepoint.com/:u:/g/personal/hzhangal_connect_ust_hk/EWgYppLPgIJApzJGkV0JY-sBM-2yDGt8dWVpowT7sjhveQ?e=9YGHXz)
3. Run the JointI model with 
```python main.py --model_type bert --model_name_or_path bert-base-uncased --output_dir your output folder --evaluate_during_training --do_train --overwrite_output_dir --model JointI```
4. You can change other settings according to the instruction in the main file.
