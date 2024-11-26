# -*- coding: utf-8 -*-
# Copyright (c) 2015-2022, Exa Analytics Development Team
# Distributed under the terms of the Apache License 2.0
"""
XYZ File Editor
##################
"""
#from __future__ import absolute_import
#from __future__ import print_function
#from __future__ import division
import six
import csv
import numpy as np
import pandas as pd
import io
#from exatomic.exa import TypedMeta
#from exatomic.exa.util.units import Length
#from exatomic.exa.util.utility import mkp
#from exatomic.core.editor import Editor
#from exatomic.core.frame import compute_frame_from_atom, Frame
#from exatomic.core.atom import Atom
#from exatomic.algorithms.indexing import starts_counts
from universe import Universe, Atom, Frame

class InputError(Exception):
    pass

#class Meta(TypedMeta):
#    atom = Atom
#    frame = Frame


class XYZ:
    """
    An editor for programmatically editing `xyz`_ files.

    .. _xyz: https://en.wikipedia.org/wiki/XYZ_file_format
    """
    _header = '{nat}\n{comment}\n'
    _cols = ['symbol', 'x', 'y', 'z']

    def parse_atom(self, unit='Angstrom', names=('symbol', 'x', 'y', 'z')):
        """
        Parse the atom table from the current xyz file.

        Args:
            unit (str): Default xyz unit of length is the Angstrom
        """
        # find where we have the number of atoms and where each
        # frame starts
        starts = []
        nats = []
        for idx, line in enumerate(self):
            if not line.strip(): continue
            try:
                nat = int(line.split()[0])
                if len(starts) != 0:
                    # check if the detected integer is in the comment line
                    if idx-1 == starts[-1]:
                        continue
                    # error out if what we expect to be the atomic position
                    # matrix has an integer instead of a string symbol
                    elif idx-2 == starts[-1]:
                        msg = "The XYZ file could not be parsed due to an error " \
                              +"with the atomic symbols. Interpreted as an " \
                              +"integer instead of a string."
                        raise InputError(msg)
                starts.append(idx)
                nats.append(nat)
            except ValueError:
                continue
        # set the rows that we want to skip
        # this would be the number of atoms and comment
        to_skip = np.concatenate((starts, np.array(starts)+1))
        comments = [self[i+1] for i in starts]
        df = pd.read_csv(six.StringIO(six.u(str(self))), delim_whitespace=True,
                                      names=names, header=None,
                                      skip_blank_lines=False,
                                      skiprows=to_skip)
        # get where the data will actually start
        initial = []
        for idx, val in enumerate(starts):
            initial.append(val-(2*idx))
        frame, _, indices = starts_counts(np.array(initial),
                                          np.array(nats))
        # some error checking
        bins = np.bincount(frame)
        for idx in np.unique(frame):
            count = bins[idx]
            if count != nats[idx]:
                raise ValueError("We had an issue with determining the number of " \
                                 +"times a frame appeared.")
        # arrange everything nicely
        df[['x', 'y', 'z']] = df[['x', 'y', 'z']].astype(np.float64)
        df['symbol'] = df['symbol'].astype('category')
        df['frame'] = frame
        df['frame'] = df['frame'].astype('category')
        df.reset_index(drop=True, inplace=True)
        df.index.names = ['atom']
        df['x'] *= 0.529177
        df['y'] *= 0.529177
        df['z'] *= 0.529177
        # add comments
        if self.meta is not None:
            self.meta['comments'] = {line: line for line in comments}
        else:
            self.meta = {'comments': {line: line for line in comments}}
        self.atom = df

    # def write(self, path, trajectory=True, float_format='%    .8f'):
    #     """
    #     Write an xyz file (or files) to disk.

    #     Args:
    #         path (str): Directory or file path
    #         trajectory (bool): Write xyz trajectory file (default) or individual

    #     Returns:
    #         path (str): On success, return the directory or file path written
    #     """
    #     if trajectory:
    #         with open(path, 'w') as f:
    #             f.write(str(self))
    #     else:
    #         grps = self.atom.cardinal_groupby()
    #         n = len(str(self.frame.index.max()))
    #         for frame, atom in grps:
    #             filename = str(frame).zfill(n) + '.xyz'
    #             with open(mkp(path, filename), 'w') as f:
    #                 f.write(self._header.format(nat=str(len(atom)),
    #                                             comment='frame: ' + str(frame)))
    #                 a = atom[self._cols].copy()
    #                 a['x'] *= Length['au', 'Angstrom']
    #                 a['y'] *= Length['au', 'Angstrom']
    #                 a['z'] *= Length['au', 'Angstrom']
    #                 a.to_csv(f, header=False, index=False, sep=' ', float_format=float_format,
    #                          quoting=csv.QUOTE_NONE, escapechar=' ')

    # @classmethod
    # def from_universe(cls, universe, atom_table='atom', float_format='%    .8f'):
    #     """
    #     Create an xyz file editor from a given universe. If the universe has
    #     more than one frame, creates an xyz trajectory format editor.

    #     Args:
    #         universe: The universe
    #         atom_table (str): One of 'atom', 'unit', or 'visual' corresponding to coordinates
    #         float_format (str): Floating point format (for writing)
    #     """
    #     string = ''
    #     grps = universe.atom.cardinal_groupby()
    #     for frame, atom in grps:
    #         string += cls._header.format(nat=len(atom), comment='frame: ' + str(frame))
    #         atom_copy = atom[cls._cols].copy()
    #         if atom_table == 'unit':
    #             atom_copy.update(universe.unit_atom)
    #         elif atom_table == 'visual':
    #             atom_copy.update(universe.visual_atom)
    #         atom_copy['x'] *= Length['au', 'Angstrom']
    #         atom_copy['y'] *= Length['au', 'Angstrom']
    #         atom_copy['z'] *= Length['au', 'Angstrom']
    #         string += atom_copy.to_csv(sep=' ', header=False, quoting=csv.QUOTE_NONE,
    #                                    index=False, float_format=float_format,
    #                                    escapechar=' ')
    #     return cls(string, name=universe.name, description=universe.description,
    #                meta=universe.meta)
    
    @classmethod
    def from_file(cls, path, **kwargs):
        """Create an editor instance from a file on disk."""
        lines = lines_from_file(path)
        if 'meta' not in kwargs:
            kwargs['meta'] = {'from': 'file'}
        kwargs['meta']['filepath'] = path
        return cls(lines, **kwargs)
    
class Editor:
    """
    Base atomic editor class for converting between file formats and to (or
    from) :class:`~exatomic.container.Universe` objects.

    Note:
        Functions defined in the editor that generate typed attributes (see
        below) should be names "parse_{data object name}".

    See Also:
        For a list of typed attributes, see :class:`~exatomic.core.universe.Universe`.
    """
    _getter_prefix = "parse"

    def parse_frame(self):
        """
        Create a minimal :class:`~exatomic.frame.Frame` from the (parsed)
        :class:`~exatomic.core.atom.Atom` object.
        """
        self.frame = compute_frame_from_atom(self.atom)

    def to_universe(self, **kws):
        """
        Convert the editor to a :class:`~exatomic.core.universe.Universe` object.

        Args:
            name (str): Name
            description (str): Description of parsed file
            meta (dict): Optional dictionary of metadata
            verbose (bool): Verbose information on failed parse methods
            ignore (bool): Ignore failed parse methods
        """
        name = kws.pop("name", None)
        description = kws.pop("description", None)
        meta = kws.pop("meta", None)
        verbose = kws.pop("verbose", True)
        ignore = kws.pop("ignore", False)
        if hasattr(self, 'meta') and self.meta is not None:
            if meta is not None:
                meta.update(self.meta)
            else:
                meta = self.meta
        kwargs = {'name': name, 'meta': meta,
                  'description': description}
        attrs = [attr.replace('parse_', '')
                 for attr in vars(self.__class__).keys()
                 if attr.startswith('parse_')]
        extras = {key: val for key, val in vars(self).items()
                  if isinstance(val, pd.DataFrame)
                  and key[1:] not in attrs}
        for attr in attrs:
            result = None
            try:
                result = getattr(self, attr)
            except Exception as e:
                if not ignore:
                    if not str(e).startswith('Please compute'):
                        print('parse_{} failed with: {}'.format(attr, e))
            if result is not None:
                kwargs[attr] = result
        kwargs.update(kws)
        kwargs.update(extras)
        return Universe(**kwargs)
    
def starts_counts(starts, counts):
    """
    Generate a pseudo-sequential array from initial values and counts.

    Args:
        starts (array): Starting points for array generation
        counts (array): Values by which to increment from each starting point

    Returns:
        arrays (tuple): First index, second index, and indices to select, respectively
    """
    n = np.sum(counts)
    i_idx = np.empty((n, ), dtype=np.int64)
    j_idx = i_idx.copy()
    values = j_idx.copy()
    h = 0
    for i, start in enumerate(starts):
        stop = start + counts[i]
        for j, value in enumerate(range(start, stop)):
            i_idx[h] = i
            j_idx[h] = j
            values[h] = value
            h += 1
    return (i_idx, j_idx, values)

def lines_from_file(path, as_interned=False, encoding=None):
    """
    Create a list of file lines from a given filepath.

    Args:
        path (str): File path
        as_interned (bool): List of "interned" strings (default False)

    Returns:
        strings (list): File line list
    """
    lines = None
    with io.open(path, encoding=encoding) as f:
        if as_interned:
            lines = [sys.intern(line) for line in f.read().splitlines()]
        else:
            lines = f.read().splitlines()
    return lines