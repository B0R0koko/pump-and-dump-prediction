"""Merge stage HOCON/JSON configuration with typed command-line overrides."""

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
import sys
from typing import Any

from pyhocon import ConfigFactory


def _config_value(action: argparse.Action, value: Any) -> Any:
    """Apply the same type and choice checks to file values as to CLI values."""
    if isinstance(action, (argparse.BooleanOptionalAction, argparse._StoreTrueAction, argparse._StoreFalseAction)):
        if not isinstance(value, bool):
            raise ValueError("expected true or false")
        return value

    def scalar(item: Any) -> Any:
        if item is None or isinstance(item, (bool, Mapping, list)):
            raise ValueError("expected a scalar value")
        if action.type is None and not isinstance(item, str):
            raise ValueError("expected a string")
        converted = (action.type or str)(str(item))
        if action.choices is not None and converted not in action.choices:
            raise ValueError(f"expected one of {list(action.choices)}")
        return converted

    if action.nargs in ("+", "*") or isinstance(action.nargs, int) and action.nargs > 0:
        if not isinstance(value, list):
            raise ValueError("expected a list")
        if action.nargs == "+" and not value:
            raise ValueError("expected a nonempty list")
        if isinstance(action.nargs, int) and len(value) != action.nargs:
            raise ValueError(f"expected {action.nargs} values")
        return [scalar(item) for item in value]
    return scalar(value)


def parse_config_args(
    parser: argparse.ArgumentParser,
    argv: Sequence[str] | None = None,
    *,
    path_keys: Sequence[str] = (),
    object_keys: Sequence[str] = (),
) -> argparse.Namespace:
    """Parse a flat stage config, then apply explicit command-line overrides.

    HOCON and JSON share the option names in ``parser``; hyphens, underscores,
    and registered aliases are accepted. Unknown keys, duplicate aliases,
    invalid types/choices and conflicting config sources are errors. File paths
    identified by ``path_keys`` resolve relative to the config, whereas paths
    passed on the CLI retain the usual current-working-directory semantics.

    ``object_keys`` allow structured, command-specific data such as pump lists;
    the caller validates their schema. ``_config_keys`` and ``_cli_keys`` expose
    destination names so callers can apply precedence to structured sources.
    """
    parser.allow_abbrev = False
    parser.add_argument("--config", type=Path, help="HOCON or JSON config file; explicit flags override its values.")
    arguments = list(sys.argv[1:] if argv is None else argv)
    if "--help" in arguments or "-h" in arguments:
        return parser.parse_args(arguments)
    probe = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    probe.add_argument("--config", type=Path)
    selected, _ = probe.parse_known_args(arguments)
    cli_keys = {
        parser._option_string_actions[token.split("=", 1)[0]].dest
        for token in arguments
        if token.split("=", 1)[0] in parser._option_string_actions
    }
    parser.set_defaults(**{key: None for key in object_keys})
    values: dict[str, Any] = {}
    config_path = None
    if selected.config is not None:
        config_path = selected.config.expanduser().resolve()
        try:
            data = ConfigFactory.parse_file(str(config_path))
        except Exception as exc:
            parser.error(f"cannot read config {config_path}: {exc}")
        if not isinstance(data, Mapping):
            parser.error("config must be an object containing option names")
        lookup = {}
        for action in parser._actions:
            if action.dest in ("help", "config"):
                continue
            lookup[action.dest.replace("-", "_")] = action
            for option in action.option_strings:
                if not option.startswith("--no-"):
                    lookup[option.lstrip("-").replace("-", "_")] = action
        for key, value in data.items():
            normalized = key.replace("-", "_")
            action = lookup.get(normalized)
            destination = action.dest if action is not None else normalized
            if action is None and normalized not in object_keys:
                parser.error(f"unknown config option: {key}")
            if destination in values:
                parser.error(f"config provides multiple aliases for {destination}")
            try:
                parsed = value if normalized in object_keys else _config_value(action, value)
                if destination in path_keys:
                    path = Path(parsed).expanduser()
                    parsed = (config_path.parent / path).resolve() if not path.is_absolute() else path.resolve()
                values[destination] = parsed
            except (TypeError, ValueError, argparse.ArgumentTypeError) as exc:
                parser.error(f"invalid config option {key}: {exc}")
        for group in parser._mutually_exclusive_groups:
            destinations = {action.dest for action in group._group_actions}
            if len(destinations.intersection(values)) > 1:
                parser.error(f"conflicting config options: {', '.join(sorted(destinations.intersection(values)))}")
            if destinations.intersection(cli_keys):
                for destination in destinations:
                    if destination not in cli_keys:
                        values.pop(destination, None)
        parser.set_defaults(**values)
    result = parser.parse_args(arguments)
    result.config = config_path
    result._config_keys = frozenset(values)
    result._cli_keys = frozenset(cli_keys)
    return result
