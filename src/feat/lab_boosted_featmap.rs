// This file is part of the open-source port of SeetaFace engine, which originally includes three modules:
//      SeetaFace Detection, SeetaFace Alignment, and SeetaFace Identification.
//
// This file is part of the SeetaFace Detection module, containing codes implementing the face detection method described in the following paper:
//
//      Funnel-structured cascade for multi-view face detection with alignment awareness,
//      Shuzhe Wu, Meina Kan, Zhenliang He, Shiguang Shan, Xilin Chen.
//      In Neurocomputing (under review)
//
// Copyright (C) 2016, Visual Information Processing and Learning (VIPL) group,
// Institute of Computing Technology, Chinese Academy of Sciences, Beijing, China.
//
// As an open-source face recognition engine: you can redistribute SeetaFace source codes
// and/or modify it under the terms of the BSD 2-Clause License.
//
// You should have received a copy of the BSD 2-Clause License along with the software.
// If not, see < https://opensource.org/licenses/BSD-2-Clause>.

use crate::ImageData;

use num::integer::Integer;
use num::traits::WrappingAdd;

use crate::common::Rectangle;
use crate::feat::FeatureMap;
use crate::math;

pub struct LabBoostedFeatureMap {
    width: u32,
    height: u32,
    length: usize,
    feat_map: Vec<u8>,
    rect_sum: Vec<i32>,
    int_img: Vec<i32>,
    square_int_img: Vec<u32>,
    rect_width: u32,
    rect_height: u32,
    num_rect: u32,
}

impl FeatureMap for LabBoostedFeatureMap {
    fn compute(&mut self, image: &ImageData) {
        let input = image.data();
        let width = image.width();
        let height = image.height();

        if width == 0 || height == 0 {
            panic!("Illegal arguments: width ({}), height ({})", width, height);
        }

        self.reshape(width, height);
        self.compute_integral_images(input);
        self.compute_rect_sum();
        self.compute_feature_map();
    }
}

impl LabBoostedFeatureMap {
    #[inline]
    pub fn new() -> Self {
        LabBoostedFeatureMap {
            width: 0,
            height: 0,
            length: 0,
            feat_map: Vec::new(),
            rect_sum: Vec::new(),
            int_img: Vec::new(),
            square_int_img: Vec::new(),
            rect_width: 3,
            rect_height: 3,
            num_rect: 3,
        }
    }

    #[inline]
    pub fn get_feature_val(&self, offset_x: i32, offset_y: i32, roi: Rectangle) -> u8 {
        let i = (roi.y() + offset_y) * (self.width as i32) + roi.x() + offset_x;
        self.feat_map[i as usize]
    }

    pub fn get_std_dev(&self, roi: Rectangle) -> f64 {
        let roi_width = roi.width() as i32;
        let roi_height = roi.height() as i32;
        let roi_x = roi.x();
        let roi_y = roi.y();
        let self_width = self.width as i32;

        let mean;
        let m2;
        let area = f64::from(roi_width * roi_height);

        match (roi_x, roi_y) {
            (0, 0) => {
                let bottom_right = (roi_height - 1) * self_width + roi_width - 1;
                mean = f64::from(self.int_img[bottom_right as usize]) / area;
                m2 = f64::from(self.square_int_img[bottom_right as usize]) / area;
            }
            (0, _) => {
                let top_right = (roi_y - 1) * self_width + roi_width - 1;
                let bottom_right = top_right + roi_height * self_width;
                mean = f64::from(
                    self.int_img[bottom_right as usize] - self.int_img[top_right as usize],
                ) / area;
                m2 = f64::from(
                    self.square_int_img[bottom_right as usize]
                        - self.square_int_img[top_right as usize],
                ) / area;
            }
            (_, 0) => {
                let bottom_left = (roi_height - 1) * self_width + roi_x - 1;
                let bottom_right = bottom_left + roi_width;
                mean = f64::from(
                    self.int_img[bottom_right as usize] - self.int_img[bottom_left as usize],
                ) / area;
                m2 = f64::from(
                    self.square_int_img[bottom_right as usize]
                        - self.square_int_img[bottom_left as usize],
                ) / area;
            }
            (_, _) => {
                let top_left = (roi_y - 1) * self_width + roi_x - 1;
                let top_right = top_left + roi_width;
                let bottom_left = top_left + roi_height * self_width;
                let bottom_right = bottom_left + roi_width;
                mean = f64::from(
                    self.int_img[bottom_right as usize] - self.int_img[bottom_left as usize]
                        + self.int_img[top_left as usize]
                        - self.int_img[top_right as usize],
                ) / area;
                m2 = f64::from(
                    self.square_int_img[bottom_right as usize]
                        .wrapping_sub(self.square_int_img[bottom_left as usize])
                        .wrapping_add(self.square_int_img[top_left as usize])
                        .wrapping_sub(self.square_int_img[top_right as usize]),
                ) / area;
            }
        }

        (m2 - mean * mean).sqrt()
    }

    fn reshape(&mut self, width: u32, height: u32) {
        self.width = width;
        self.height = height;
        self.length = width as usize * height as usize;

        self.feat_map.resize(self.length, 0);
        self.rect_sum.resize(self.length, 0);
        self.int_img.resize(self.length, 0);
        self.square_int_img.resize(self.length, 0);
    }

    fn compute_integral_images(&mut self, input: &[u8]) {
        assert_eq!(input.len(), self.length);

        math::copy_u8_to_i32(input, &mut self.int_img);
        math::square(&self.int_img, &mut self.square_int_img);

        LabBoostedFeatureMap::compute_integral(
            &mut self.int_img,
            self.width,
            self.height,
        );
        LabBoostedFeatureMap::compute_integral(
            &mut self.square_int_img,
            self.width,
            self.height,
        );
    }

    fn compute_integral<T: Integer + WrappingAdd + Copy>(
        data: &mut [T],
        width: u32,
        height: u32,
    ) {
        let mut data_iter_mut = data.iter_mut().take(width as usize);
        let mut src = data_iter_mut.next().unwrap();
        for dest in data_iter_mut {
            *dest = *src + *dest;
            src = dest;
        }

        let mut row_iter = data.chunks_exact_mut(width as usize).take(height as usize);
        let mut previous_row = row_iter.next().unwrap();
        for row in row_iter {
            let mut s: T = num::zero();
            for (dest_previous_row, dest) in previous_row.iter().zip(row.iter_mut()) {
                s = s + *dest;
                // overflow does happen here for the list of squares..
                // original code does not seem to worry about this though
                *dest = (*dest_previous_row).wrapping_add(&s);
            }
            previous_row = row;
        }
    }

    fn compute_rect_sum(&mut self) {
        let width = (self.width - self.rect_width) as usize;
        let height = self.height - self.rect_height;

        let int_img = self.int_img.as_slice();
        let rect_sum = self.rect_sum.as_mut_slice();

        rect_sum[0] = int_img[((self.rect_height - 1) * self.width + self.rect_width - 1) as usize];
        math::vector_sub_safe(
            &int_img[((self.rect_height - 1) * self.width + self.rect_width) as usize..],
            &int_img[((self.rect_height - 1) * self.width) as usize..],
            &mut rect_sum[1..][..width],
        );

        for i in 1..(height + 1) {
            let top_left = &int_img[((i - 1) * self.width) as usize..];
            let top_right = &top_left[(self.rect_width - 1) as usize..];
            let bottom_left = &top_left[(self.rect_height * self.width) as usize..];
            let bottom_right = &bottom_left[(self.rect_width - 1) as usize..];

            let mut dest = &mut rect_sum[(i * self.width) as usize..];
            dest[0] = bottom_right[0] - top_right[0];
            dest = &mut dest[1..][..width];

            math::vector_sub_safe(&bottom_right[1..], &top_right[1..], dest);
            math::vector_calculation(bottom_left, dest, |bottom_left, dest| dest - bottom_left);
            math::vector_calculation(top_left, dest, |top_left, dest| dest + top_left);
        }
    }

    fn compute_feature_map(&mut self) {
        let width = self.width - self.rect_width * self.num_rect;
        let height = self.height - self.rect_height * self.num_rect;
        let offset = self.width * self.rect_height;

        for r in 0..(height + 1) {
            for c in 0..(width + 1) {
                let dest = &mut self.feat_map[(r * self.width + c) as usize];
                *dest = 0;

                let white_rect_sum = self.rect_sum
                    [((r + self.rect_height) * self.width + c + self.rect_width) as usize];

                let mut black_rect_idx = r * self.width + c;
                if white_rect_sum >= self.rect_sum[black_rect_idx as usize] {
                    *dest |= 0x80
                };

                black_rect_idx += self.rect_width;
                if white_rect_sum >= self.rect_sum[black_rect_idx as usize] {
                    *dest |= 0x40
                };
                black_rect_idx += self.rect_width;
                if white_rect_sum >= self.rect_sum[black_rect_idx as usize] {
                    *dest |= 0x20
                };

                black_rect_idx += offset;
                if white_rect_sum >= self.rect_sum[black_rect_idx as usize] {
                    *dest |= 0x08
                };
                black_rect_idx += offset;
                if white_rect_sum >= self.rect_sum[black_rect_idx as usize] {
                    *dest |= 0x01
                };

                black_rect_idx -= self.rect_width;
                if white_rect_sum >= self.rect_sum[black_rect_idx as usize] {
                    *dest |= 0x02
                };
                black_rect_idx -= self.rect_width;
                if white_rect_sum >= self.rect_sum[black_rect_idx as usize] {
                    *dest |= 0x04
                };

                black_rect_idx -= offset;
                if white_rect_sum >= self.rect_sum[black_rect_idx as usize] {
                    *dest |= 0x10
                };
            }
        }
    }
}
