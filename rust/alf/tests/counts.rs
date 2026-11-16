use alf::{read_alf_bytes, read_alf_header_bytes, Channel};

fn envelope(header: &str, payload: &[u8]) -> Vec<u8> {
    let mut bytes = vec![1];
    bytes.extend_from_slice(&(header.len() as u32).to_le_bytes());
    bytes.extend_from_slice(header.as_bytes());
    bytes.extend_from_slice(payload);
    bytes
}

// Also executed as WebAssembly by the runtime example. A 64-bit native build
// cannot detect truncation of a header count to a 32-bit usize.
pub fn verify_counts() {
    for token in [
        "4294967296",
        "18446744073709551616",
        "18446744073709551616.0",
        "18446744073709551616e0",
    ] {
        let expected = token.parse::<f64>().unwrap();
        let bytes = envelope(&format!(r#"{{"_total_points":{token}}}"#), &[]);
        let head = read_alf_header_bytes(&bytes, Channel::Ali).unwrap();
        assert_eq!(head.total_points, Some(expected));
        assert_eq!(
            read_alf_bytes(&bytes, Channel::Ali).unwrap_err().id,
            "payload-length-mismatch"
        );
        let sidecar = envelope(
            &format!(r#"{{"_total_points":1,"_source_total_points":{token}}}"#),
            &[0, 0, 0, 0],
        );
        let file = read_alf_bytes(&sidecar, Channel::Alx).unwrap();
        assert_eq!(file.kept_indices, vec![0]);
        assert_eq!(
            file.metadata["_source_total_points"].as_f64(),
            Some(expected)
        );
    }
    for token in ["4294967296", "4611686018427387904"] {
        let bytes = envelope(
            &format!(r#"{{"_total_points":{token},"_source_total_points":1}}"#),
            &[],
        );
        assert_eq!(
            read_alf_bytes(&bytes, Channel::Alx).unwrap_err().id,
            "payload-length-mismatch"
        );
    }
}

#[test]
fn counts_do_not_depend_on_machine_integer_width() {
    verify_counts();
}
