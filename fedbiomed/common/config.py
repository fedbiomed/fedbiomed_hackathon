# This file is originally part of Fed-BioMed
# SPDX-License-Identifier: Apache-2.0

import os
import uuid
import pathlib
import configparser

from abc import ABCMeta, abstractmethod

from fedbiomed.common.constants import (
    MPSPDZ_certificate_prefix,
    VAR_FOLDER_NAME,
    TMP_FOLDER_NAME,
    CACHE_FOLDER_NAME,
    DB_PREFIX,
)
from fedbiomed.common.utils import (
    raise_for_version_compatibility,
)
from fedbiomed.common.certificate_manager import generate_certificate
from fedbiomed.common.exceptions import FedbiomedError


CONFIG_FILE_NAME = "config.ini"


class Config(metaclass=ABCMeta):
    """Base Config class"""

    _CONFIG_VERSION: str

    def __init__(self, config: str, auto_generate: bool = True) -> None:
        """Initializes configuration"""

        # First try to get component specific config file name, then CONFIG_FILE
        self._config_file = config
        self._cfg = configparser.ConfigParser()

        if auto_generate:
            self.generate()

    @classmethod
    @abstractmethod
    def _COMPONENT_TYPE(cls):  # pylint: disable=C0103
        """Abstract attribute to oblige defining component type"""

    @classmethod
    @abstractmethod
    def _CONFIG_VERSION(cls):  # pylint: disable=C0103
        """Abstract attribute to oblige defining component type"""

    def is_config_existing(self) -> bool:
        """Checks if config file exists

        Returns:
            True if config file is already existing
        """

        return os.path.isfile(self._config_file)

    def read(self) -> bool:
        """Reads configuration file that is already existing in given path

        Raises verision compatibility error
        """

        self._cfg.read(self._config_file)

        # Validate config version
        raise_for_version_compatibility(
            self._cfg["default"]["version"],
            self._CONFIG_VERSION,
            f"Configuration file {self.path}: found version %s expected version %s",
        )

        return True

    def get(self, section, key, **kwargs) -> str:
        """Returns value for given key and section"""

        return self._cfg.get(section, key, **kwargs)

    def set(self, section, key, value) -> None:
        """Sets configuration section values

        Args:
            section: the name of the configuration file section as defined
                by the `.ini` standard
            key: the name of the attribute to be set
            value: the value of the attribute to be set

        Returns:
            value: the value of the attribute that was just set
        """
        self._cfg.set(section, key, value)

    def sections(self) -> list:
        """Returns sections of the config"""

        return self._cfg.sections()

    def write(self) -> None:
        """Writes config file"""

        with open(self._config_file, "w", encoding="UTF-8") as f:
            self._cfg.write(f)

    def generate(self, force: bool = False, component_id: str | None = None) -> bool:
        """ "Generate configuration file

        Args:
            force: Overwrites existing configuration file
            id: Component ID
        """

        # Check if configuration is already existing
        if self.is_config_existing() and not force:
            return self.read()

        component_root = os.path.dirname(self._config_file)

        # Create default section
        component_id = (
            component_id if component_id else f"{self._COMPONENT_TYPE}_{uuid.uuid4()}"
        )

        self._cfg["default"] = {
            "id": component_id,
            "component": self._COMPONENT_TYPE,
            "version": str(self._CONFIG_VERSION),
        }

        db_path = os.path.join(component_root, f"{DB_PREFIX}{component_id}.json")

        self._cfg["default"]["db"] = db_path

        # Generate self-signed certificates
        key_file, pem_file = generate_certificate(
            path=os.path.join(component_root, "certs"),
            component_id=component_id,
            prefix=MPSPDZ_certificate_prefix,
        )

        self._cfg["mpspdz"] = {
            "private_key": key_file,
            "public_key": pem_file,
        }

        # Calls child class add_parameters
        self.add_parameters()

        # Write configuration file
        return self.write()

    @abstractmethod
    def add_parameters(self):
        """ "Component specific argument creation"""

    def refresh(self):
        """Refreshes config file by recreating all the fields without
        chaning component ID.
        """

        if not self.is_config_existing():
            raise FedbiomedError("Can not refresh config file that is not existing")

        # Read the config
        self._cfg.read(self._config_file)
        component_id = self._cfg["default"]["id"]

        # Generate by keeping the component ID
        self.generate(force=True, component_id=component_id)


class Component:

    _COMPONENT_TYPE: str
    _CONFIG_CLASS: Config
    _FOLDERS: list[str] = []

    def __init__(self, path: str | None = None) -> None:

        # If there is path provided use it otherwise it is the working directory
        self._path = os.path.abspath(path) if path else os.getcwd()
        self._config = self._CONFIG_CLASS(conf=os.path.join(path, CONFIG_FILE_NAME))

    def create(self) -> None:
        """Creates component folder and instantiate a config object"""

        if not os.path.isdir(self._path):
            os.makedirs(self._path)
        else:
            self._validate_root_path()

        var_dir = os.path.join(self._path, VAR_FOLDER_NAME)
        cache_dir = os.path.join(var_dir, CACHE_FOLDER_NAME)
        tmp_dir = os.path.join(var_dir, TMP_FOLDER_NAME)

        for folder in [*self._FOLDERS, var_dir, cache_dir, tmp_dir]:
            pathlib.Path(os.path.join(self._path, folder)).mkdir(exist_ok=True)

    def _validate_root_path(self) -> None:
        """Validate if a new component can be created in given path"""

        fedbiomed_com = os.path.join(self._path, ".fedbiomed-component")
        if os.path.isfile(fedbiomed_com):
            with open(fedbiomed_com, "r", encoding="UTF-8") as file:
                component = file.read()
                if component != self._COMPONENT_TYPE:
                    raise FedbiomedError(
                        f"There is a different component already instatiated in the given "
                        f"component root path {self._path}. Component already instaiated "
                        f"{component}, can not create component {self._COMPONENT_TYPE} "
                    )
        else:
            with open(fedbiomed_com, "w", encoding="UTF-8") as file:
                file.write(f"{self._COMPONENT_TYPE}")
