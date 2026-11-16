use alf::{read_alf_bytes, AlfFile, Channel};

fn geometry(version: u8, fields: &str, words: &[u16]) -> AlfFile {
    let header = format!(r#"{{"_total_points":{}{fields}}}"#, words.len() / 3);
    let mut bytes = vec![version];
    bytes.extend_from_slice(&(header.len() as u32).to_le_bytes());
    bytes.extend_from_slice(header.as_bytes());
    for word in words {
        bytes.extend_from_slice(&word.to_le_bytes());
    }
    read_alf_bytes(&bytes, Channel::Alp).unwrap()
}

fn fixed(scale: f64, offset: [f64; 3], recorded: Option<f64>, words: &[u16]) -> AlfFile {
    let fields = format!(
        r#", "_geom_codec":"int16", "_frame_convention":"world-aligned", "_geom_scale":{scale:e}, "_geom_offset":{offset:?}{}"#,
        recorded
            .map(|value| format!(r#", "_geom_max_err_m":{value:e}"#))
            .unwrap_or_default(),
    );
    geometry(3, &fields, words)
}

#[cfg_attr(test, test)]
pub fn fp16_largest_finite_component_has_a_finite_bound() {
    let file = geometry(1, "", &[0x7bff, 0xfbff, 0]);
    assert_eq!(file.xyz, vec![[65504.0, -65504.0, 0.0]]);
    assert_eq!(file.max_geom_error_m, Some(16.0));
}

#[cfg_attr(test, test)]
pub fn fp16_all_invalid_components_keep_the_subnormal_floor() {
    let file = geometry(1, "", &[0x7e00, 0x7c00, 0xfc00]);
    assert!(file.xyz[0].iter().all(|value| !value.is_finite()));
    assert_eq!(file.max_geom_error_m, Some(2f64.powi(-25)));
    for words in [[0, 0, 0], [1, 0x8001, 0]] {
        assert_eq!(
            geometry(1, "", &words).max_geom_error_m,
            Some(2f64.powi(-25))
        );
    }
    assert_eq!(geometry(1, "", &[]).max_geom_error_m, Some(0.0));
}

#[cfg_attr(test, test)]
pub fn empty_fixed_point_geometry_has_no_error() {
    let file = fixed(0.0001, [0.0; 3], Some(7.0), &[]);
    assert!(file.xyz.is_empty());
    assert_eq!(file.max_geom_error_m, Some(0.0));
}

#[cfg_attr(test, test)]
pub fn subnormal_half_steps_never_round_below_the_exact_half() {
    for units in 1u64..=8 {
        let scale = f64::from_bits(units);
        for recorded in [None, Some(0.0)] {
            let bound = fixed(scale, [0.0; 3], recorded, &[0, 0, 0])
                .max_geom_error_m
                .unwrap();
            // In units of the minimum subnormal, ceil(units/2) is the least
            // representable conservative half-step, independent of host math.
            let minimum = f64::from_bits(units.div_ceil(2));
            assert!(bound >= minimum, "scale={scale:e}, bound={bound:e}");
            assert!(bound <= f64::from_bits(units.div_ceil(2) + 1));
        }
    }
}

#[cfg_attr(test, test)]
pub fn largest_finite_product_keeps_its_reconstruction_allowance() {
    let file = fixed(f64::MAX, [0.0; 3], None, &[1, 0, 0]);
    assert_eq!(file.xyz, vec![[f64::MAX, 0.0, 0.0]]);
    // MAX/2 plus half its finite adjacent spacing rounds to 2^1023.
    assert_eq!(
        file.max_geom_error_m,
        Some(f64::from_bits(0x7fe0000000000000))
    );
    let exact_offset = fixed(0.0001, [f64::MAX; 3], None, &[0, 0, 0]);
    assert_eq!(exact_offset.max_geom_error_m, Some(0.00005));
}

pub fn verify_geometry_bounds() {
    fp16_largest_finite_component_has_a_finite_bound();
    fp16_all_invalid_components_keep_the_subnormal_floor();
    empty_fixed_point_geometry_has_no_error();
    subnormal_half_steps_never_round_below_the_exact_half();
    largest_finite_product_keeps_its_reconstruction_allowance();
}
