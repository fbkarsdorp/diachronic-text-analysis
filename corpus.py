# Hierarchical Agglomerative Cluster Analysis
#
# Copyright (C) 2011 Institute for Dutch Lexicology (INL)
# Author: Folgert Karsdorp <folgert.karsdorp@inl.nl>
# URL: <http://www.inl.nl/>
# For licence information, see LICENCE.TXT

import string

class Text(object):
    def __init__(self, id, title, author, year, text, kind=None):
        self._id = id
        self._text = text
        self._title = title
        self._author = author
        self._year = int(year)
        self._kind = kind
        # remove all punctuation, transform to lowercase and split on spaces
        self._tokens = text.translate(None, string.punctuation).lower().split()

    def __hash__(self): # IS THIS ALLOWED?
        return self._id

    def title(self): return self._title
    def author(self): return self._author
    def year(self): return self._year
    def kind(self): return self._kind

    def __getitem__(self, i):
        if isinstance(i, slice):
            return self._tokens[i.start:i.stop]
        else:
            return self._tokens[i]

    def __len__(self):
        return len(self._tokens)

