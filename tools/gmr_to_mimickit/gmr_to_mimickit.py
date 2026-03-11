"""
This module provides functionality to convert motion data from GMR format to MimicKit format. 

Usage:
    Command line:
        python tools/data_format/gmr_to_mimickit.py
    Required arguments (choose one mode):
        [single file mode]
        --input_file PATH       Path to the input GMR pickle file
        --output_file PATH      Path to save the output MimicKit pickle file

        [batch folder mode]
        --input_folder PATH     Path to a folder containing GMR pickle files
        --output_folder PATH    Path to a folder to save converted MimicKit pickle files
    Optional arguments:
        --loop {wrap,clamp}     Loop mode for the motion (default: wrap)
        --start_frame INT       Start frame for motion clipping (default: 0)
        --end_frame INT         End frame for motion clipping (default: -1, uses all frames)
        --output_fps INT        Frame rate for the output motion (default: same as input)
    
GMR Format:
    The input GMR format should be a pickle file containing a dictionary with keys:
    - 'fps': Frame rate (int)
    - 'root_pos': Root position array, shape (num_frames, 3)
    - 'root_rot': Root rotation quaternions, shape (num_frames, 4), format (x, y, z, w)
    - 'dof_pos': Degrees of freedom positions, shape (num_frames, num_dofs)
    - 'local_body_pos': Currently unused (can be None)
    - 'link_body_list': Currently unused (can be None)

Output:
    Creates a dictionary containing MimicKit motion data saved as a pickle file, with loop mode stored as INT and motion data stored as
    concatenated arrays of [root_pos, root_rot_expmap, dof_pos] per frame.
"""

import argparse
import os
import pickle
import numpy as np
import sys

import torch

sys.path.append(".")  # Ensure the repository root is on sys.path so we can use some utilities.

from mimickit.anim.motion import Motion, LoopMode
from mimickit.util.torch_util import quat_to_exp_map


def _register_numpy_pickle_compat_aliases():
    """
    Register module aliases to improve pickle compatibility between NumPy versions.

    Some pickle files created in a different NumPy version may reference
    internal module paths like `numpy._core.*` or `numpy.core.*`.
    """
    try:
        import numpy.core as np_core

        # NumPy<2 may not expose numpy._core, but pickles from NumPy>=2 can reference it.
        sys.modules.setdefault("numpy._core", np_core)

        if hasattr(np_core, "multiarray"):
            sys.modules.setdefault("numpy._core.multiarray", np_core.multiarray)
        if hasattr(np_core, "numeric"):
            sys.modules.setdefault("numpy._core.numeric", np_core.numeric)
    except Exception:
        # Best-effort compatibility shim.
        pass


def _load_pickle_compat(file_path):
    """
    Load pickle with compatibility fallback for NumPy internal module paths.
    """
    try:
        with open(file_path, "rb") as f:
            return pickle.load(f)
    except ModuleNotFoundError as e:
        # Common cross-version error: No module named 'numpy._core'
        if "numpy._core" in str(e):
            _register_numpy_pickle_compat_aliases()
            with open(file_path, "rb") as f:
                return pickle.load(f)
        raise

def convert_gmr_to_mimickit(gmr_file_path, output_file_path, loop_mode, start_frame, end_frame, output_fps):
    """
    Convert a GMR compatible motion dataset to MimicKit compatible dataset.
    
    Args:
        gmr_file_path (str): Path to the GMR format pickle file
        output_file_path (str): Path to save the MimicKit format pickle file
        loop_mode (bool): Whether the motion should loop (Set to wrap as default)
    """
    if loop_mode == "wrap":
        loop_mode_out = LoopMode.WRAP # MimicKit LoopMode.WRAP
    elif loop_mode == "clamp":
        loop_mode_out = LoopMode.CLAMP # MimicKit LoopMode.CLAMP
    else:
        raise ValueError(f"Invalid loop_mode: {loop_mode}. Choose 'wrap' or 'clamp'.")
    
    # Load GMR format data
    gmr_data = _load_pickle_compat(gmr_file_path)
    
    # Extract data from GMR format
    fps = gmr_data['fps']
    root_pos = gmr_data['root_pos']  # Shape: (num_frames, 3)
    root_rot_quat = gmr_data['root_rot']  # Shape: (num_frames, 4), quaternion format
    dof_pos = gmr_data['dof_pos']    # Shape: (num_frames, num_dofs)

    # Log the type and shape of each extracted term
    print("\n" + "="*60)
    print("📥 LOADED GMR DATA")
    print("="*60)
    print(f"⏱️  FPS:           type={type(fps).__name__}, value={fps}")
    print(f"📍 Root Position: type={type(root_pos).__name__}, shape={root_pos.shape}")
    print(f"🔄 Root Rotation: type={type(root_rot_quat).__name__}, shape={root_rot_quat.shape}")
    print(f"🦴 DOF Position:  type={type(dof_pos).__name__}, shape={dof_pos.shape}")
    print("="*60 + "\n")
    
    # Verify shapes
    if root_pos.ndim != 2 or root_pos.shape[1] != 3:
        raise ValueError(f"Expected root_pos shape (num_frames, 3), got {root_pos.shape}")
        
    if root_rot_quat.ndim != 2 or root_rot_quat.shape[1] != 4:
        raise ValueError(f"Expected root_rot_quat shape (num_frames, 4), got {root_rot_quat.shape}")
        
    if dof_pos.ndim != 2:
        raise ValueError(f"Expected dof_pos to be 2D array, got {dof_pos.ndim}D")

    # Convert quaternion to exponential map
    root_rot = quat_to_exp_map(torch.tensor(root_rot_quat)).numpy()  
    
    # Stack all motion data along the last dimension
    # frames shape: (num_frames, 3 + 3 + num_dofs) = (num_frames, 6 + num_dofs)
    frames = np.concatenate([root_pos, root_rot, dof_pos], axis=1)

    # Chop frames
    if end_frame == -1:
        end_frame = frames.shape[0]
    assert 0 <= start_frame < end_frame <= frames.shape[0], "Invalid start_frame or end_frame."
    frames = frames[start_frame:end_frame, :]

    save_fps = fps if output_fps == -1 else output_fps

    out_data = Motion(loop_mode=loop_mode_out, fps=save_fps, frames=frames)

    # Save to MimicKit format
    out_data.save(output_file_path)
    
    print("\n" + "="*60)
    print("✅ CONVERSION SUCCESSFUL")
    print("="*60)
    print(f"📁 Input:  {gmr_file_path}")
    print(f"💾 Output: {output_file_path}")
    print("-"*60)
    print(f"📊 Frames Shape:  {frames.shape}")
    print(f"🎬 Total Frames: {frames.shape[0]}")
    print(f"⏱️  FPS:          {save_fps}")
    print(f"🔄 Loop Mode:    {loop_mode_out}")
    print("="*60 + "\n")

    return out_data


def convert_gmr_folder_to_mimickit(input_folder, output_folder, loop_mode, start_frame, end_frame, output_fps):
    """
    Convert all .pkl files in a folder from GMR format to MimicKit format.

    Args:
        input_folder (str): Directory containing input GMR pickle files
        output_folder (str): Directory to save converted MimicKit pickle files
    """
    if not os.path.isdir(input_folder):
        raise ValueError(f"Input folder does not exist or is not a directory: {input_folder}")

    os.makedirs(output_folder, exist_ok=True)

    input_files = sorted(
        [file_name for file_name in os.listdir(input_folder) if file_name.lower().endswith(".pkl")]
    )

    if len(input_files) == 0:
        raise ValueError(f"No .pkl files found in input folder: {input_folder}")

    print("\n" + "=" * 60)
    print("🚀 BATCH CONVERSION START")
    print("=" * 60)
    print(f"📂 Input Folder:  {input_folder}")
    print(f"📂 Output Folder: {output_folder}")
    print(f"📦 Files to convert: {len(input_files)}")
    print("=" * 60 + "\n")

    success_count = 0
    failure_files = []

    for idx, input_name in enumerate(input_files, start=1):
        input_path = os.path.join(input_folder, input_name)
        output_path = os.path.join(output_folder, input_name)

        print(f"[{idx}/{len(input_files)}] Converting: {input_name}")

        try:
            convert_gmr_to_mimickit(
                input_path,
                output_path,
                loop_mode=loop_mode,
                start_frame=start_frame,
                end_frame=end_frame,
                output_fps=output_fps,
            )
            success_count += 1
        except Exception as e:
            print(f"❌ Failed: {input_name} -> {e}")
            failure_files.append((input_name, str(e)))

    print("\n" + "=" * 60)
    print("📌 BATCH CONVERSION SUMMARY")
    print("=" * 60)
    print(f"✅ Succeeded: {success_count}/{len(input_files)}")
    print(f"❌ Failed:    {len(failure_files)}/{len(input_files)}")

    if failure_files:
        print("-" * 60)
        for file_name, err in failure_files:
            print(f"• {file_name}: {err}")

    print("=" * 60 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Convert GMR motion data to MimicKit format.")
    parser.add_argument("--input_file", help="Path to the input GMR pickle file")
    parser.add_argument("--output_file", help="Path to the output MimicKit pickle file")
    parser.add_argument("--input_folder", help="Path to a folder containing input GMR pickle files")
    parser.add_argument("--output_folder", help="Path to a folder for output MimicKit pickle files")
    parser.add_argument("--loop", default="wrap", choices=["wrap", "clamp"], help="Enable loop mode on the converted motion")
    parser.add_argument("--start_frame", type=int, default=0, help="Start frame for chopping the motion")
    parser.add_argument("--end_frame", type=int, default=-1, help="End frame for chopping the motion")
    parser.add_argument("--output_fps", type=int, default=-1, help="Frame rate for the output motion (default: same as input)")
    args = parser.parse_args()

    single_file_mode = args.input_file is not None or args.output_file is not None
    folder_mode = args.input_folder is not None or args.output_folder is not None

    if single_file_mode and folder_mode:
        parser.error("Use either file mode (--input_file/--output_file) or folder mode (--input_folder/--output_folder), not both.")

    if single_file_mode:
        if not args.input_file or not args.output_file:
            parser.error("In file mode, both --input_file and --output_file are required.")

        convert_gmr_to_mimickit(
            args.input_file,
            args.output_file,
            loop_mode=args.loop,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
            output_fps=args.output_fps,
        )
    elif folder_mode:
        if not args.input_folder or not args.output_folder:
            parser.error("In folder mode, both --input_folder and --output_folder are required.")

        convert_gmr_folder_to_mimickit(
            args.input_folder,
            args.output_folder,
            loop_mode=args.loop,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
            output_fps=args.output_fps,
        )
    else:
        parser.error("Provide either file mode arguments or folder mode arguments.")


if __name__ == "__main__":
    main()