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

use crate::common::{Rectangle, Seq};
use crate::feat::FeatureMap;
use crate::math;
use crate::ImageData;
use std::ops::Deref;

#[cfg(feature = "rayon")]
use rayon::prelude::*;

pub struct SurfMlpFeatureMap {
    width: u32,
    height: u32,
    length: usize,
    feature_pool: FeaturePool,
    feature_vectors: Vec<Vec<i32>>,
    feature_vectors_normalized: Vec<Vec<f32>>,
    grad_x: Vec<i32>,
    grad_y: Vec<i32>,
    int_img: Vec<i32>,
    img_buf: Vec<i32>,
}

impl FeatureMap for SurfMlpFeatureMap {
    fn compute(&mut self, image: &ImageData) {
        let input = image.data();
        let width = image.width();
        let height = image.height();

        if width == 0 || height == 0 {
            panic!("Illegal arguments: width ({}), height ({})", width, height);
        }

        self.reshape(width, height);
        self.compute_gradient_images(input);
        self.compute_integral_images();
    }
}

impl SurfMlpFeatureMap {
    pub fn new() -> Self {
        let feature_pool = SurfMlpFeatureMap::create_feature_pool();
        let feature_pool_size = feature_pool.size();
        let mut feature_vectors = Vec::with_capacity(feature_pool_size);
        let mut feature_vectors_normalized = Vec::with_capacity(feature_pool_size);
        for feature_id in 0..feature_pool_size {
            let dim = feature_pool.get_feature_vector_dim(feature_id);
            feature_vectors.push(vec![0; dim]);
            feature_vectors_normalized.push(vec![0.0; dim]);
        }

        SurfMlpFeatureMap {
            width: 0,
            height: 0,
            length: 0,
            feature_pool,
            feature_vectors,
            feature_vectors_normalized,
            grad_x: Vec::new(),
            grad_y: Vec::new(),
            int_img: Vec::new(),
            img_buf: Vec::new(),
        }
    }

    fn create_feature_pool() -> FeaturePool {
        let mut feature_pool = FeaturePool::new();
        feature_pool.add_patch_format(1, 1, 2, 2);
        feature_pool.add_patch_format(1, 2, 2, 2);
        feature_pool.add_patch_format(2, 1, 2, 2);
        feature_pool.add_patch_format(2, 3, 2, 2);
        feature_pool.add_patch_format(3, 2, 2, 2);
        feature_pool.create();
        feature_pool
    }

    fn reshape(&mut self, width: u32, height: u32) {
        self.width = width;
        self.height = height;
        self.length = width as usize * height as usize;

        self.grad_x.resize(self.length, 0);
        self.grad_y.resize(self.length, 0);
        self.int_img
            .resize(self.length * FeaturePool::K_NUM_INT_CHANNEL as usize, 0);
        self.img_buf.resize(self.length, 0);
    }

    fn compute_gradient_images(&mut self, input: &[u8]) {
        assert_eq!(input.len(), self.length);

        math::copy_u8_to_i32(input, &mut self.img_buf);
        self.compute_grad_x();
        self.compute_grad_y();
    }

    fn compute_grad_x(&mut self) {
        for (src_row, dest_row) in self.img_buf.chunks_exact_mut(self.width as usize)
            .zip(self.grad_x.chunks_exact_mut(self.width as usize)) {

            let mut src_iter = src_row.iter().copied();
            let src_first = src_iter.next().unwrap();
            let src_second = src_iter.next().unwrap();
            *dest_row.first_mut().unwrap() = (src_second - src_first) << 1;

            let (_, src_offset_2) = src_row.split_at(2);
            let (_, dest_offset_1) = dest_row.split_at_mut(1);
            math::vector_sub(src_offset_2, src_row, dest_offset_1);

            let mut src_iter = src_row.iter().copied();
            let src_last = src_iter.next_back().unwrap();
            let src_second_last = src_iter.next_back().unwrap();
            *dest_row.last_mut().unwrap() = (src_last - src_second_last) << 1;
        }
    }

    fn compute_grad_y(&mut self) {
        let mut src_row_iter = self.img_buf.chunks_exact(self.width as usize);
        let src_row_first = src_row_iter.next().unwrap();
        let src_row_second = src_row_iter.next().unwrap();
        math::vector_sub(src_row_second, src_row_first, &mut self.grad_y);
        math::vector_mul(&mut self.grad_y[..self.width as usize], 2);

        let row_len = self.width as usize;
        let last_row_start = self.grad_y.len() - row_len;

        #[cfg(feature = "rayon")]
        let iter = self.img_buf.par_chunks_exact(row_len)
            .zip(self.img_buf.par_chunks_exact(row_len).skip(2))
            .zip(self.grad_y[row_len..last_row_start].par_chunks_exact_mut(row_len));

        #[cfg(not(feature = "rayon"))]
        let iter = self.img_buf.chunks_exact(row_len)
            .zip(self.img_buf.chunks_exact(row_len).skip(2))
            .zip(self.grad_y[row_len..last_row_start].chunks_exact_mut(row_len));

        iter.for_each(|((src_row, src_row_next_twice), dest)| {
            math::vector_sub(src_row_next_twice, src_row, dest);
        });

        let mut src_row_iter = self.img_buf.chunks_exact(self.width as usize);
        let src_row_last = src_row_iter.next_back().unwrap();
        let src_row_second_last = src_row_iter.next_back().unwrap();
        let dest_row_last = self.grad_y.chunks_exact_mut(self.width as usize).last().unwrap();
        math::vector_sub(
            src_row_last,
            src_row_second_last,
            dest_row_last,
        );
        math::vector_mul(dest_row_last, 2);
    }

    fn compute_integral_images(&mut self) {
        Self::fill_integral_channel(&self.grad_x, &mut self.int_img, 0);
        Self::fill_integral_channel(&self.grad_y, &mut self.int_img, 4);
        math::abs(&self.grad_x, &mut self.img_buf);
        Self::fill_integral_channel(&self.img_buf, &mut self.int_img, 1);
        math::abs(&self.grad_y, &mut self.img_buf);
        Self::fill_integral_channel(&self.img_buf, &mut self.int_img, 5);

        self.mask_integral_channel();
        self.integral();
    }

    fn fill_integral_channel(src: &[i32], int_img: &mut [i32], ch: u32) {
        for (src_value, dest_vec) in src.iter().zip(int_img.chunks_exact_mut(FeaturePool::K_NUM_INT_CHANNEL as usize)) {
            dest_vec[ch as usize] = *src_value;
            dest_vec[ch as usize + 2] = *src_value;
        }
    }

    fn mask_integral_channel(&mut self) {
        let xor_bits: [u32; 4] = [0xffff_ffff, 0xffff_ffff, 0, 0];
        for ((dx, dy), int_img_vec) in self.grad_x.iter().copied()
            .zip(self.grad_y.iter().copied())
            .zip(self.int_img.chunks_exact_mut(FeaturePool::K_NUM_INT_CHANNEL as usize)) {

            let (first, second) = int_img_vec.split_at_mut(4);

            let cmp = if dy < 0 { 0xffff_ffff } else { 0x0 };
            for (dest, j) in first.iter_mut().zip(xor_bits.iter()) {
                let dy_mask = (cmp ^ j) as i32;
                *dest &= dy_mask;
            }

            let cmp = if dx < 0 { 0xffff_ffff } else { 0x0 };
            for (dest, j) in second.iter_mut().zip(xor_bits.iter()) {
                let dx_mask = (cmp ^ j) as i32;
                *dest &= dx_mask;
            }
        }
    }

    fn integral(&mut self) {
        let len = (FeaturePool::K_NUM_INT_CHANNEL * self.width) as usize;

        let mut row_iter = self.int_img.chunks_exact_mut(len);
        if let Some(mut row1) = row_iter.next() {
            for row2 in row_iter {
                math::vector_add(row1, row2);
                SurfMlpFeatureMap::vector_cumulative_add(
                    row1,
                    FeaturePool::K_NUM_INT_CHANNEL,
                );
                row1 = row2;
            }
            SurfMlpFeatureMap::vector_cumulative_add(
                row1,
                FeaturePool::K_NUM_INT_CHANNEL,
            );
        }
    }

    #[inline]
    fn vector_cumulative_add(x: &mut[i32], num_channel: u32) {
        SurfMlpFeatureMap::vector_cumulative_add_portable(x, num_channel);
    }

    fn vector_cumulative_add_portable(x: &mut[i32], num_channel: u32) {
        let mut col_iter = x.chunks_exact_mut(num_channel as usize);
        if let Some(mut col1) = col_iter.next() {
            for col2 in col_iter {
                math::vector_add(col1, col2);
                col1 = col2;
            }
        }
    }

    fn compute_feature_vector(&mut self, feature_id: usize, roi: Rectangle) {
        let feature = self.feature_pool.get_feature(feature_id);

        let init_cell_x = roi.x() + feature.patch.x();
        let init_cell_y = roi.y() + feature.patch.y();
        let k_num_int_channel = FeaturePool::K_NUM_INT_CHANNEL as isize;
        let cell_width: isize =
            (feature.patch.width() / feature.num_cell_per_row) as isize * k_num_int_channel;
        let cell_height: isize = (feature.patch.height() / feature.num_cell_per_col) as isize;
        let row_width: isize = (self.width as isize) * k_num_int_channel;

        let mut cell_top_left = [Cursor::empty(); FeaturePool::K_NUM_INT_CHANNEL as usize];
        let mut cell_top_right = [Cursor::empty(); FeaturePool::K_NUM_INT_CHANNEL as usize];
        let mut cell_bottom_left = [Cursor::empty(); FeaturePool::K_NUM_INT_CHANNEL as usize];
        let mut cell_bottom_right = [Cursor::empty(); FeaturePool::K_NUM_INT_CHANNEL as usize];

        let mut dest_iter = self.feature_vectors[feature_id].iter_mut();
        let int_img = self.int_img.as_slice();

        match (init_cell_x, init_cell_y) {
            (0, 0) => {
                let mut offset: isize = row_width * (cell_height - 1) + cell_width - k_num_int_channel;
                for i in 0..k_num_int_channel as usize {
                    cell_bottom_right[i] = Cursor::with_offset(int_img, offset);
                    offset += 1;
                    *dest_iter.next().unwrap() = *cell_bottom_right[i];
                    cell_top_right[i] = cell_bottom_right[i];
                }

                for _ in 1..feature.num_cell_per_row {
                    for j in 0..k_num_int_channel as usize {
                        cell_bottom_left[j] = cell_bottom_right[j];
                        cell_bottom_right[j] = cell_bottom_right[j].offset(cell_width);
                        *dest_iter.next().unwrap() = *cell_bottom_right[j] - *cell_bottom_left[j];
                    }
                }
            }
            (_, 0) => {
                let mut offset: isize =
                    row_width * (cell_height - 1) + (init_cell_x - 1) as isize * k_num_int_channel;
                for i in 0..k_num_int_channel as usize {
                    cell_bottom_left[i] = Cursor::with_offset(int_img, offset);
                    offset += 1;
                    cell_bottom_right[i] = cell_bottom_left[i].offset(cell_width);
                    *dest_iter.next().unwrap() = *cell_bottom_right[i] - *cell_bottom_left[i];
                    cell_top_right[i] = cell_bottom_right[i];
                }

                for _ in 1..feature.num_cell_per_row {
                    for j in 0..k_num_int_channel as usize {
                        cell_bottom_left[j] = cell_bottom_right[j];
                        cell_bottom_right[j] = cell_bottom_right[j].offset(cell_width);
                        *dest_iter.next().unwrap() = *cell_bottom_right[j] - *cell_bottom_left[j];
                    }
                }
            }
            (0, _) => {
                let mut tmp_cell_top_right: Vec<Cursor<'_>> =
                    vec![Cursor::empty(); k_num_int_channel as usize];

                let mut offset: isize = row_width * ((init_cell_y - 1) as isize) + cell_width - k_num_int_channel;
                for i in 0..k_num_int_channel as usize {
                    cell_top_right[i] = Cursor::with_offset(int_img, offset);
                    offset += 1;
                    cell_bottom_right[i] = cell_top_right[i].offset(row_width * cell_height);
                    tmp_cell_top_right[i] = cell_bottom_right[i];
                    *dest_iter.next().unwrap() = *cell_bottom_right[i] - *cell_top_right[i];
                }

                for _ in 1..feature.num_cell_per_row {
                    for j in 0..k_num_int_channel as usize {
                        cell_top_left[j] = cell_top_right[j];
                        cell_top_right[j] = cell_top_right[j].offset(cell_width);
                        cell_bottom_left[j] = cell_bottom_right[j];
                        cell_bottom_right[j] = cell_bottom_right[j].offset(cell_width);
                        *dest_iter.next().unwrap() = *cell_bottom_right[j] + *cell_top_left[j]
                            - *cell_top_right[j]
                            - *cell_bottom_left[j];
                    }
                }

                cell_top_right[..k_num_int_channel as usize]
                    .clone_from_slice(&tmp_cell_top_right[..k_num_int_channel as usize]);
            }
            (_, _) => {
                let mut tmp_cell_top_right: Vec<Cursor<'_>> =
                    vec![Cursor::empty(); k_num_int_channel as usize];

                let mut offset: isize = row_width * ((init_cell_y - 1) as isize)
                    + (init_cell_x - 1) as isize * k_num_int_channel;
                for i in 0..k_num_int_channel as usize {
                    cell_top_left[i] = Cursor::with_offset(int_img, offset);
                    offset += 1;
                    cell_top_right[i] = cell_top_left[i].offset(cell_width);
                    cell_bottom_left[i] = cell_top_left[i].offset(row_width * cell_height);
                    cell_bottom_right[i] = cell_bottom_left[i].offset(cell_width);
                    *dest_iter.next().unwrap() = *cell_bottom_right[i] + *cell_top_left[i]
                        - *cell_top_right[i]
                        - *cell_bottom_left[i];
                    tmp_cell_top_right[i] = cell_bottom_right[i];
                }

                for _ in 1..feature.num_cell_per_row {
                    for j in 0..k_num_int_channel as usize {
                        cell_top_left[j] = cell_top_right[j];
                        cell_top_right[j] = cell_top_right[j].offset(cell_width);
                        cell_bottom_left[j] = cell_bottom_right[j];
                        cell_bottom_right[j] = cell_bottom_right[j].offset(cell_width);
                        *dest_iter.next().unwrap() = *cell_bottom_right[j] + *cell_top_left[j]
                            - *cell_top_right[j]
                            - *cell_bottom_left[j];
                    }
                }

                cell_top_right[..k_num_int_channel as usize]
                    .clone_from_slice(&tmp_cell_top_right[..k_num_int_channel as usize]);
            }
        }

        let offset: isize = cell_height * row_width - feature.patch.width() as isize * k_num_int_channel
            + cell_width;
        for _ in 1..feature.num_cell_per_row {
            if init_cell_x == 0 {
                for j in 0..k_num_int_channel as usize {
                    cell_bottom_right[j] = cell_bottom_right[j].offset(offset);
                    *dest_iter.next().unwrap() = *cell_bottom_right[j] - *cell_top_right[j];
                }
            } else {
                for j in 0..k_num_int_channel as usize {
                    cell_bottom_right[j] = cell_bottom_right[j].offset(offset);
                    cell_top_left[j] = cell_top_right[j].offset(-cell_width);
                    cell_bottom_left[j] = cell_bottom_right[j].offset(-cell_width);
                    *dest_iter.next().unwrap() = *cell_bottom_right[j] + *cell_top_left[j]
                        - *cell_top_right[j]
                        - *cell_bottom_left[j];
                }
            }

            for _ in 1..feature.num_cell_per_row {
                for k in 0..k_num_int_channel as usize {
                    cell_top_left[k] = cell_top_right[k];
                    cell_top_right[k] = cell_top_right[k].offset(cell_width);
                    cell_bottom_left[k] = cell_bottom_right[k];
                    cell_bottom_right[k] = cell_bottom_right[k].offset(cell_width);
                    *dest_iter.next().unwrap() = *cell_bottom_right[k] + *cell_top_left[k]
                        - *cell_bottom_left[k]
                        - *cell_top_right[k];
                }
            }

            for j in cell_top_right.iter_mut().take(k_num_int_channel as usize) {
                *j = j.offset(offset);
            }
        }
    }

    pub fn get_feature_vector(
        &mut self,
        feature_id: usize,
        feature_vec: &mut [f32],
        roi: Rectangle,
    ) {
        self.compute_feature_vector(feature_id, roi);

        SurfMlpFeatureMap::normalize_feature_vector(
            &self.feature_vectors[feature_id],
            &mut self.feature_vectors_normalized[feature_id],
        );

        let feature_vec_normalized = self.feature_vectors_normalized[feature_id].as_slice();
        let dest = &mut feature_vec[..feature_vec_normalized.len()];
        dest.copy_from_slice(feature_vec_normalized);
    }

    fn normalize_feature_vector(feature_vec: &[i32], feature_vec_normalized: &mut [f32]) {
        let prod: f64 = feature_vec
            .iter()
            .copied()
            .map(|value| f64::from(value * value))
            .sum();

        if prod != 0.0 {
            let norm = prod.sqrt() as f32;
            for (dst, src) in feature_vec_normalized.iter_mut().zip(feature_vec) {
                *dst = *src as f32 / norm;
            }
        } else {
            for dst in feature_vec_normalized {
                *dst = 0.0;
            }
        }
    }

    #[inline]
    pub fn get_feature_vector_dim(&self, feature_id: usize) -> usize {
        self.feature_pool.get_feature_vector_dim(feature_id)
    }
}

#[derive(Clone, Copy)]
struct Cursor<'a> {
    all_data: &'a [i32],
    offset: isize,
    data: i32,
}

impl <'a> Cursor<'a> {
    fn with_offset(all_data: &'a [i32], offset: isize) -> Self {
        Self {
            all_data,
            offset,
            data: all_data[offset as usize],
        }
    }

    fn empty() -> Self {
        Self {
            all_data: &[],
            offset: 0,
            data: 0,
        }
    }

    fn offset(&self, offset: isize) -> Self {
        let new_offset = self.offset + offset;
        let data = self.all_data[new_offset as usize];
        Self {
            all_data: self.all_data,
            offset: new_offset,
            data,
        }
    }
}

impl Deref for Cursor<'_> {
    type Target = i32;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

struct FeaturePool {
    sample_width: u32,
    sample_height: u32,
    patch_move_step_x: u32,
    patch_move_step_y: u32,
    patch_size_inc_step: u32,
    patch_min_width: u32,
    patch_min_height: u32,
    features: Vec<Feature>,
    patch_formats: Vec<PatchFormat>,
}

impl FeaturePool {
    const K_NUM_INT_CHANNEL: u32 = 8;

    #[inline]
    fn new() -> Self {
        FeaturePool {
            sample_width: 40,
            sample_height: 40,
            patch_move_step_x: 16,
            patch_move_step_y: 16,
            patch_size_inc_step: 1,
            patch_min_width: 16,
            patch_min_height: 16,
            features: Vec::new(),
            patch_formats: Vec::new(),
        }
    }

    fn add_patch_format(
        &mut self,
        width: u32,
        height: u32,
        num_cell_per_row: u32,
        num_cell_per_col: u32,
    ) {
        self.patch_formats.push(PatchFormat {
            width,
            height,
            num_cell_per_row,
            num_cell_per_col,
        });
    }

    fn create(&mut self) {
        let mut feature_vecs = Vec::new();

        if self.sample_height - self.patch_min_height <= self.sample_width - self.patch_min_width {
            for format in &self.patch_formats {
                for h in Seq::new(self.patch_min_height, |x| x + self.patch_size_inc_step)
                    .take_while(|x| *x <= self.sample_height)
                {
                    if h % format.num_cell_per_col != 0 || h % format.height != 0 {
                        continue;
                    }
                    let w = h / format.height * format.width;
                    if w % format.num_cell_per_row != 0
                        || w < self.patch_min_width
                        || w > self.sample_width
                    {
                        continue;
                    }
                    self.collect_features(
                        w,
                        h,
                        format.num_cell_per_row,
                        format.num_cell_per_col,
                        &mut feature_vecs,
                    );
                }
            }
        } else {
            for format in &self.patch_formats {
                // original condition was <= self.patch_min_width,
                // but it would not make sense to have a loop in such case
                for w in Seq::new(self.patch_min_width, |x| x + self.patch_size_inc_step)
                    .take_while(|x| *x <= self.sample_width)
                {
                    if w % format.num_cell_per_row != 0 || w % format.width != 0 {
                        continue;
                    }
                    let h = w / format.width * format.height;
                    if h % format.num_cell_per_col != 0
                        || h < self.patch_min_height
                        || h > self.sample_height
                    {
                        continue;
                    }
                    self.collect_features(
                        w,
                        h,
                        format.num_cell_per_row,
                        format.num_cell_per_col,
                        &mut feature_vecs,
                    );
                }
            }
        }

        self.features.append(&mut feature_vecs);
    }

    fn collect_features(
        &self,
        width: u32,
        height: u32,
        num_cell_per_row: u32,
        num_cell_per_col: u32,
        dest: &mut Vec<Feature>,
    ) {
        let y_lim = self.sample_height - height;
        let x_lim = self.sample_width - width;

        for y in Seq::new(0, |n| n + self.patch_move_step_y).take_while(|n| *n <= y_lim) {
            for x in Seq::new(0, |n| n + self.patch_move_step_x).take_while(|n| *n <= x_lim) {
                dest.push(Feature {
                    patch: Rectangle::new(x as i32, y as i32, width, height),
                    num_cell_per_row,
                    num_cell_per_col,
                });
            }
        }
    }

    #[inline]
    fn size(&self) -> usize {
        self.features.len()
    }

    #[inline]
    fn get_feature(&self, feature_id: usize) -> &Feature {
        &self.features[feature_id]
    }

    #[inline]
    fn get_feature_vector_dim(&self, feature_id: usize) -> usize {
        let feature = &self.features[feature_id];
        (feature.num_cell_per_col * feature.num_cell_per_row * FeaturePool::K_NUM_INT_CHANNEL)
            as usize
    }
}

struct PatchFormat {
    width: u32,
    height: u32,
    num_cell_per_row: u32,
    num_cell_per_col: u32,
}

struct Feature {
    patch: Rectangle,
    num_cell_per_row: u32,
    num_cell_per_col: u32,
}
