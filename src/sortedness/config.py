#  Copyright (c) 2023. Davi Pereira dos Santos
#  This file is part of the sortedness project.
#  Please respect the license - more about this in the section (*) below.
#
#  sortedness is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  sortedness is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with sortedness.  If not, see <http://www.gnu.org/licenses/>.
#
#  (*) Removing authorship by any means, e.g. by distribution of derived
#  works or verbatim, obfuscated, compiled or rewritten versions of any
#  part of this work is illegal and it is unethical regarding the effort and
#  time spent here.
#
from configparser import ConfigParser
from pathlib import Path

config = ConfigParser()
config.read(f"{Path.home()}/.cache.config")
try:  # pragma: no cover
    local_cache_uri = config.get("Storage", "local")
    near_cache_uri = config.get("Storage", "near")
    remote_cache_uri = config.get("Storage", "remote")
    schedule_uri = config.get("Storage", "schedule")
except Exception as e:
    print(
        "Please create a config file '.cache.config' in your home folder following the template:\n"
        """[Storage]
local = sqlite+pysqlite:////home/davi/.hdict.cache.db
remote = mysql+pymysql://username:xxxxxxxxxxxxxxxxxxxxx@url/database

[Path]
images_dir = /tmp/"""
    )
