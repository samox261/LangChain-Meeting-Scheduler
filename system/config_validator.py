import yaml
import os
import json
from typing import Dict, Any, List, Tuple
import logging
from datetime import datetime
import pytz

class ConfigValidator:
    REQUIRED_FIELDS = {
        "agent_email_address": str,
        "timezone": str,
        "poll_interval_seconds": int,
        "assistant_mode_settings": dict,
        "preferred_meeting_durations": list
    }

    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.config = {}
        self.errors = []
        self.warnings = []

    def validate_credentials_file(self) -> bool:
        """Validate the credentials.json file exists and has required fields."""
        if not os.path.exists("credentials.json"):
            self.errors.append("Missing credentials.json file")
            return False
        
        try:
            with open("credentials.json", 'r') as f:
                creds = json.load(f)
                if "installed" not in creds:
                    self.errors.append("Missing 'installed' section in credentials.json")
                    return False
                
                installed = creds["installed"]
                required_fields = ["client_id", "client_secret", "project_id", "auth_uri", "token_uri"]
                for field in required_fields:
                    if field not in installed:
                        self.errors.append(f"Missing required field '{field}' in credentials.json installed section")
                return all(field in installed for field in required_fields)
        except json.JSONDecodeError:
            self.errors.append("Invalid JSON in credentials.json")
            return False
        except Exception as e:
            self.errors.append(f"Error reading credentials.json: {str(e)}")
            return False

    def validate(self) -> Tuple[bool, List[str], List[str]]:
        """Validate the configuration file.
        Returns: (is_valid, errors, warnings)"""
        try:
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        except Exception as e:
            self.errors.append(f"Failed to load config file: {str(e)}")
            return False, self.errors, self.warnings

        # Check required fields
        for field, field_type in self.REQUIRED_FIELDS.items():
            if field not in self.config:
                self.errors.append(f"Missing required field: {field}")
            elif not isinstance(self.config[field], field_type):
                self.errors.append(f"Invalid type for {field}: expected {field_type.__name__}, got {type(self.config[field]).__name__}")

        # Validate email address
        if "agent_email_address" in self.config:
            email = self.config["agent_email_address"]
            if not isinstance(email, str) or "@" not in email or "." not in email:
                self.errors.append("Invalid agent_email_address format")

        # Validate timezone
        if "timezone" in self.config:
            try:
                pytz.timezone(self.config["timezone"])
            except pytz.exceptions.UnknownTimeZoneError:
                self.errors.append(f"Invalid timezone: {self.config['timezone']}")

        # Validate poll interval
        if "poll_interval_seconds" in self.config:
            interval = self.config["poll_interval_seconds"]
            if not isinstance(interval, int) or interval < 30:
                self.warnings.append("poll_interval_seconds should be at least 30 seconds")

        # Validate meeting durations
        if "preferred_meeting_durations" in self.config:
            durations = self.config["preferred_meeting_durations"]
            if not isinstance(durations, list) or not durations:
                self.errors.append("preferred_meeting_durations must be a non-empty list")
            else:
                for duration in durations:
                    if not isinstance(duration, int) or duration <= 0:
                        self.errors.append("All meeting durations must be positive integers")

        # Validate assistant mode settings
        if "assistant_mode_settings" in self.config:
            settings = self.config["assistant_mode_settings"]
            if not isinstance(settings, dict):
                self.errors.append("assistant_mode_settings must be a dictionary")
            else:
                if "authorized_command_senders" not in settings:
                    self.warnings.append("No authorized_command_senders specified in assistant_mode_settings")
                elif not isinstance(settings["authorized_command_senders"], list):
                    self.errors.append("authorized_command_senders must be a list")
                else:
                    for email in settings["authorized_command_senders"]:
                        if not isinstance(email, str) or "@" not in email or "." not in email:
                            self.errors.append(f"Invalid email in authorized_command_senders: {email}")

        # Validate credentials.json instead of environment variables
        self.validate_credentials_file()

        return len(self.errors) == 0, self.errors, self.warnings

    def get_validation_report(self) -> str:
        """Generate a human-readable validation report."""
        is_valid, errors, warnings = self.validate()
        
        report = []
        report.append(f"Configuration Validation Report - {datetime.now().isoformat()}")
        report.append("=" * 50)
        report.append(f"Overall Status: {'VALID' if is_valid else 'INVALID'}")
        report.append("\nErrors:")
        if errors:
            for error in errors:
                report.append(f"  - {error}")
        else:
            report.append("  None")
        
        report.append("\nWarnings:")
        if warnings:
            for warning in warnings:
                report.append(f"  - {warning}")
        else:
            report.append("  None")
        
        return "\n".join(report)

if __name__ == "__main__":
    validator = ConfigValidator()
    print(validator.get_validation_report()) 