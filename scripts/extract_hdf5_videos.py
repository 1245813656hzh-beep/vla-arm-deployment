#!/usr/bin/env python3
"""
从 HDF5 数据集中提取相机视频文件

用法:
    python extract_hdf5_videos.py --input /path/to/dataset.hdf5 --output ./videos

功能:
    - 提取 HDF5 中所有相机的图像数据
    - 为每个 demo 的每个相机生成 MP4 视频
    - 支持批量处理多个 demos
"""

import h5py
import cv2
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm


def extract_videos_from_hdf5(
    hdf5_path: str, output_dir: str = "./videos", fps: int = 30, demo_indices: list = None
):
    """从 HDF5 文件中提取所有相机的视频

    Args:
        hdf5_path: HDF5 文件路径
        output_dir: 输出视频目录
        fps: 视频帧率
        demo_indices: 指定要处理的 demo 索引列表，None 表示处理所有
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with h5py.File(hdf5_path, "r") as f:
        if "data" not in f:
            print(f"错误: HDF5 文件中没有 'data' 组")
            return

        demo_keys = sorted(f["data"].keys(), key=lambda x: int(x.split("_")[-1]))

        if demo_indices is not None:
            demo_keys = [demo_keys[i] for i in demo_indices if i < len(demo_keys)]

        print(f"找到 {len(demo_keys)} 个 demos")
        print(f"输出目录: {output_path}")
        print("-" * 60)

        for demo_name in tqdm(demo_keys, desc="处理 demos"):
            demo_path = f"data/{demo_name}/obs"

            if demo_path not in f:
                print(f"警告: {demo_name} 没有 obs 数据，跳过")
                continue

            obs_group = f[demo_path]

            # 查找相机数据 (4维数组: 帧数 x 高 x 宽 x 通道)
            cameras = []
            for key in obs_group.keys():
                ds = obs_group[key]
                if hasattr(ds, "shape") and len(ds.shape) == 4 and ds.dtype == "uint8":
                    cameras.append(key)

            if not cameras:
                print(f"警告: {demo_name} 中没有找到相机数据")
                continue

            print(f"\n{demo_name}: 找到 {len(cameras)} 个相机: {cameras}")

            # 为每个相机创建视频
            for cam_name in cameras:
                frames = obs_group[cam_name][:]
                num_frames, height, width, channels = frames.shape

                # 创建视频文件路径
                video_filename = f"{demo_name}_{cam_name}.mp4"
                video_path = output_path / video_filename

                # 创建视频写入器
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                out = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))

                if not out.isOpened():
                    print(f"错误: 无法创建视频文件 {video_path}")
                    continue

                # 逐帧写入
                for frame_idx in range(num_frames):
                    frame = frames[frame_idx]
                    # RGB -> BGR (OpenCV 需要 BGR 格式)
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    out.write(frame_bgr)

                out.release()
                print(f"  ✓ {cam_name}: {num_frames} 帧 -> {video_filename}")

    print("\n" + "=" * 60)
    print(f"完成! 所有视频已保存到: {output_path}")


def extract_single_camera(
    hdf5_path: str, camera_name: str, output_dir: str = "./videos", fps: int = 30
):
    """只提取指定相机的视频

    Args:
        hdf5_path: HDF5 文件路径
        camera_name: 相机名称 (如 'table_cam', 'wrist_cam')
        output_dir: 输出目录
        fps: 帧率
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with h5py.File(hdf5_path, "r") as f:
        demo_keys = sorted(f["data"].keys(), key=lambda x: int(x.split("_")[-1]))

        print(f"提取相机 '{camera_name}' 从 {len(demo_keys)} 个 demos")

        for demo_name in tqdm(demo_keys, desc="处理"):
            cam_path = f"data/{demo_name}/obs/{camera_name}"

            if cam_path not in f:
                print(f"警告: {demo_name} 中没有 {camera_name}，跳过")
                continue

            frames = f[cam_path][:]
            num_frames, height, width, _ = frames.shape

            video_path = output_path / f"{demo_name}_{camera_name}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))

            for frame_idx in range(num_frames):
                frame_bgr = cv2.cvtColor(frames[frame_idx], cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)

            out.release()
            print(f"  ✓ {demo_name}: {num_frames} 帧")

    print(f"\n完成! 视频保存到: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="从 HDF5 数据集中提取相机视频",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 提取所有相机的视频
  python extract_hdf5_videos.py --input dataset.hdf5 --output ./videos
  
  # 只提取 table_cam
  python extract_hdf5_videos.py --input dataset.hdf5 --camera table_cam
  
  # 只处理前 5 个 demo
  python extract_hdf5_videos.py --input dataset.hdf5 --demos 0 1 2 3 4
        """,
    )

    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default="/home/intern/vla-arm-deployment/datasets/franka_place_bin_3camera_augmented.hdf5",
        help="输入 HDF5 文件路径",
    )

    parser.add_argument(
        "--output", "-o", type=str, default="./videos", help="输出视频目录 (默认: ./videos)"
    )

    parser.add_argument(
        "--camera",
        "-c",
        type=str,
        default=None,
        help="只提取指定相机 (如: table_cam, wrist_cam, table_cam_side)",
    )

    parser.add_argument(
        "--demos",
        type=int,
        nargs="+",
        default=None,
        help="指定 demo 索引列表 (如: 0 1 2)，默认处理所有",
    )

    parser.add_argument("--fps", type=int, default=30, help="视频帧率 (默认: 30)")

    args = parser.parse_args()

    # 检查输入文件
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"错误: 输入文件不存在: {args.input}")
        return

    print(f"HDF5 文件: {args.input}")
    print(f"输出目录: {args.output}")
    print(f"FPS: {args.fps}")
    print("=" * 60)

    if args.camera:
        # 只提取指定相机
        extract_single_camera(args.input, args.camera, args.output, args.fps)
    else:
        # 提取所有相机
        extract_videos_from_hdf5(args.input, args.output, args.fps, args.demos)


if __name__ == "__main__":
    main()
