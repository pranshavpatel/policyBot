# Multi-Factor Authentication (MFA) Policy

See also VPN Reset for MFA during a VPN password reset specifically, and Password Policy for password rules.

## 1. Where MFA Is Required
- MFA is required for: SSO login, VPN connections, email access from a new device, and any system holding Confidential or Restricted data (see Data Security Policy).

## 2. Approved MFA Methods
- Authenticator app (preferred) — supports offline codes.
- SMS text code — available as a fallback, but discouraged due to SIM-swap risk.
- Hardware security key — required for employees with access to Restricted data.

## 3. Enrollment
- MFA enrollment is required during onboarding, before your account is activated (see Onboarding Policy).
- At least one backup method must be enrolled in addition to your primary method.

## 4. Lost Device / Lost Access
- If you lose access to your MFA method (lost phone, etc.), contact IT via `#it-support` with your employee ID for identity verification and a temporary bypass code.
- Temporary bypass codes expire after 24 hours and require re-enrollment of a permanent MFA method.

## 5. Frequency
- MFA is required once per new device/browser per 30 days ("remember this device"), except for Restricted-data systems, which require MFA on every login.
