#/bin/sh

# We want netcdf as an optional dependency but stdeb makes it
# a hard dependency. Let's change this.

echo Repacking .deb file with altered dependencies
DEBFILE=`ls deb_dist | grep deb$`
DEBFILE=deb_dist/${DEBFILE}
echo DEBFILE IS ${DEBFILE}

TMP=tmp_dir
touch ${TMP}
rm -rf ${TMP}
mkdir ${TMP}
dpkg-deb -x ${DEBFILE} ${TMP}
dpkg-deb --control ${DEBFILE} ${TMP}/DEBIAN
sed -i 's/python-netcdf4, //' ${TMP}/DEBIAN/control
sed -i 's/Suggests: /Suggests: python-netcdf4, /' ${TMP}/DEBIAN/control
dpkg -b ${TMP} ${DEBFILE}
rm -rf ${TMP}
