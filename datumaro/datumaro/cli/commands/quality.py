
# Copyright (C) 2020 Intel Corporation
#
# SPDX-License-Identifier: MIT

import argparse
import json
import logging as log
import os
import os.path as osp

from datumaro.components.launcher import ModelTransform
from datumaro.components.operations import DistanceComparator

from ..contexts.project import compute_ann_statistics
from ..contexts.project.diff import DiffVisualizer
from ..util import CliException, MultilineFormatter
from ..util.project import generate_next_file_name, load_project


def build_parser(parser_ctor=argparse.ArgumentParser):
    parser = parser_ctor(help="",
        description="""

        """,
        formatter_class=MultilineFormatter)

    parser.add_argument('-m', '--model',
        help="Model to be used for inference")
    parser.add_argument('--inference-path', default=None,
        help="Path to a project with inference")
    parser.add_argument('-f', '--format',
        default=DiffVisualizer.DEFAULT_FORMAT,
        choices=[f.name for f in DiffVisualizer.Format],
        help="Output format (default: %(default)s)")
    parser.add_argument('--iou-thresh', default=0.5, type=float,
        help="IoU match threshold for detections (default: %(default)s)")
    parser.add_argument('--conf-thresh', default=0.75, type=float,
        help="Confidence threshold for detections (default: %(default)s)")
    parser.add_argument('-o', '--output-dir', default=None, dest='dst_dir',
        help="Save directory for output")
    parser.add_argument('--overwrite', action='store_true',
        help="Overwrite existing files in the save directory")
    parser.add_argument('-p', '--project', dest='project_dir', default='.',
        help="Directory of the project to operate on (default: current dir)")
    parser.set_defaults(command=quality_command)

    return parser

def quality_command(args):
    project = load_project(args.project_dir)

    dst_dir = args.dst_dir
    if dst_dir:
        if not args.overwrite and osp.isdir(dst_dir) and os.listdir(dst_dir):
            raise CliException("Directory '%s' already exists "
                "(pass --overwrite to force creation)" % dst_dir)
    else:
        dst_dir = generate_next_file_name('%s-quality-report' % \
            project.config.project_name)
    dst_dir = osp.abspath(dst_dir)

    log.info("Loading dataset...")
    dataset = project.make_dataset()

    if args.model:
        log.info("Running inference...")

        model = project.make_executable_model(args.model)
        # something is needed to filter produced annotations by confidence
        # to improve performance and lower noise

        inference = dataset.transform(ModelTransform, launcher=model)
    elif args.inference_path:
        inference = project.load(args.inference_path).make_dataset()

    comparator = Comparator(iou_threshold=args.iou_thresh,
        conf_threshold=args.conf_thresh)
    visualizer = DiffVisualizer(save_dir=dst_dir, comparator=comparator,
        output_format=args.format)
    visualizer.save_dataset_diff(inference, dataset)

    stats = compute_ann_statistics(dataset)
    with open(generate_next_file_name(
            'statistics', basedir=dst_dir, ext='.json'), 'w') as f:
        json.dump(stats, f, indent=4, sort_keys=True)

    log.info("Dataset quality report saved to '%s'" % dst_dir)

    return 0
