# Hierarchical Agglomerative Cluster Analysis
#
# Copyright (C) 2013 Folgert Karsdorp
# Author: Folgert Karsdorp <fbkarsdorp@gmail.com>
# URL: <https://github.com/fbkarsdorp/HAC-python>
# For licence information, see LICENCE.TXT

def ngrams(text, n):
    """Return N grams of a text."""
    count = max(0, len(text) - n + 1)
    return (tuple(text[i:i+n]) for i in xrange(count))


