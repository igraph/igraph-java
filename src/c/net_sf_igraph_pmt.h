/* 
   IGraph library Java interface.
   Copyright (C) 2007  Tamas Nepusz <ntamas@rmki.kfki.hu>
   MTA RMKI, Konkoly-Thege Miklos st. 29-33, Budapest 1121, Hungary
   
   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 2 of the License, or
   (at your option) any later version.
   
   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.
   
   You should have received a copy of the GNU General Public License
   along with this program; if not, write to the Free Software
   Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 
   02110-1301 USA 

*/

/*

ATTENTION: This is a highly experimental, proof-of-concept Java interface.
Its main purpose was to convince me that it can be done in finite time :)
The interface is highly incomplete, at the time of writing even some
essential functions (e.g. addEdges) are missing. Since I don't use Java
intensively, chances are that this interface gets finished only if there
is substantial demand for it and/or someone takes the time to send patches
or finish it completely.

*/

#ifndef CONCAT2
#  define CONCAT2x(a,b)     a ## _ ## b
#  define CONCAT2(a,b)      CONCAT2x(a,b)
#endif

#ifndef CONCAT3
#  define CONCAT3x(a,b,c)   a ## _ ## b ## _ ## c
#  define CONCAT3(a,b,c)    CONCAT3x(a,b,c)
#endif

#ifndef CONCAT4
#  define CONCAT4x(a,b,c,d) a ## _ ## b ## _ ## c ## _ ## d
#  define CONCAT4(a,b,c,d)  CONCAT4x(a,b,c)
#endif

#define FUNCTION(a) CONCAT3(Java_net_sf_igraph, JAVA_TYPE, a)
