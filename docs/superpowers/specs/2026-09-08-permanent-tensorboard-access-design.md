# Permanent reward-redesign TensorBoard access (2026-09-08)

**Status:** APPROVED by user

## Goal

Keep `http://127.0.0.1:16008` as the stable local URL for the 30 formal reward-redesign runs across machine reboots and transient SSH/server failures, without exposing TensorBoard to the public network.

## Design

- Store the curated 30-run symlink view under the persistent server result tree rather than `/tmp`.
- Run TensorBoard on server loopback port `16007`; a guarded launcher prevents duplicate instances.
- Add an idempotent user `@reboot` cron entry for the server launcher.
- Register a Windows per-user logon task that opens local port `16008` to server loopback port `16007` through SSH.
- Configure the Windows task to restart after failure and ignore duplicate task instances.

## Verification

- Confirm the server launcher is active and its cron entry exists.
- Confirm the Windows scheduled task is registered and running.
- Query the local TensorBoard API and require exactly the expected 30 formal run names.
- Confirm every run exposes `train/episode_reward`.

## Scope

The URL is persistent on this Windows computer after user login. It is intentionally not a public or cross-device hosted endpoint.
