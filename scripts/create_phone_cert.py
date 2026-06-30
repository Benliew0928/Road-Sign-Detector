from __future__ import annotations

import argparse
import socket
from datetime import UTC, datetime, timedelta
from ipaddress import ip_address
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID


def _local_ipv4_addresses() -> list[str]:
    candidates: set[str] = set()
    try:
        for result in socket.getaddrinfo(socket.gethostname(), None, family=socket.AF_INET):
            address = str(result[4][0])
            parsed = ip_address(address)
            if not parsed.is_loopback and not parsed.is_link_local:
                candidates.add(address)
    except OSError:
        pass
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect(("10.255.255.255", 1))
            address = str(probe.getsockname()[0])
            parsed = ip_address(address)
            if not parsed.is_loopback and not parsed.is_link_local:
                candidates.add(address)
    except OSError:
        pass
    return sorted(candidates)


def _subject_alt_names(hosts: list[str]) -> x509.SubjectAlternativeName:
    names: list[x509.GeneralName] = []
    for host in dict.fromkeys(hosts):
        try:
            names.append(x509.IPAddress(ip_address(host)))
        except ValueError:
            names.append(x509.DNSName(host))
    return x509.SubjectAlternativeName(names)


def create_certificate(cert_path: Path, key_path: Path, hosts: list[str], days: int) -> None:
    cert_path.parent.mkdir(parents=True, exist_ok=True)
    key_path.parent.mkdir(parents=True, exist_ok=True)

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = issuer = x509.Name(
        [
            x509.NameAttribute(NameOID.COMMON_NAME, "RoadSign Assist local phone camera"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "RoadSign Assist local development"),
        ]
    )
    now = datetime.now(UTC)
    certificate = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=5))
        .not_valid_after(now + timedelta(days=days))
        .add_extension(_subject_alt_names(hosts), critical=False)
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .sign(key, hashes.SHA256())
    )

    key_path.write_bytes(
        key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    cert_path.write_bytes(certificate.public_bytes(serialization.Encoding.PEM))


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a local HTTPS cert for phone-camera testing.")
    parser.add_argument("--cert", default="certs/roadsign-local.crt")
    parser.add_argument("--key", default="certs/roadsign-local.key")
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--hosts", nargs="*", default=[])
    args = parser.parse_args()

    hosts = ["localhost", "127.0.0.1", socket.gethostname(), *_local_ipv4_addresses(), *args.hosts]
    create_certificate(Path(args.cert), Path(args.key), hosts, args.days)
    print(f"Certificate: {Path(args.cert).resolve()}")
    print(f"Private key:  {Path(args.key).resolve()}")
    print("Subject alternative names:")
    for host in dict.fromkeys(hosts):
        print(f"  - {host}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
