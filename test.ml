<?xml version='1.0' encoding='utf-8'?>
<q:quakeml xmlns:q="http://quakeml.org/xmlns/quakeml/1.2" xmlns="http://quakeml.org/xmlns/bed/1.2">
  <eventParameters publicID="smi:local/59c8f863-8ef7-43fa-84ff-410e22c60efc">
    <event publicID="smi:local/ndk/C201708270417A/event">
      <preferredOriginID>smi:local/ndk/C201708270417A/origin#cmtorigin</preferredOriginID>
      <preferredMagnitudeID>smi:local/ndk/C201708270417A/magnitude#moment_mag</preferredMagnitudeID>
      <preferredFocalMechanismID>smi:local/ndk/C201708270417A/focal_mechanism</preferredFocalMechanismID>
      <type>earthquake</type>
      <typeCertainty>known</typeCertainty>
      <description>
        <text>ADMIRALTY ISLANDS REGION, P.N.G.</text>
        <type>Flinn-Engdahl region</type>
      </description>
      <description>
        <text>C201708270417A</text>
        <type>earthquake name</type>
      </description>
      <origin publicID="smi:local/ndk/C201708270417A/origin#reforigin">
        <time>
          <value>2017-08-27T04:17:51.000000Z</value>
        </time>
        <latitude>
          <value>-1.45</value>
        </latitude>
        <longitude>
          <value>148.08</value>
        </longitude>
        <depth>
          <value>8000.0</value>
        </depth>
        <type>hypocenter</type>
        <comment id="smi:local/ndk/C201708270417A/comment#ref_origin">
          <text>Hypocenter catalog: PDEW</text>
        </comment>
      </origin>
      <origin publicID="smi:local/ndk/C201708270417A/origin#cmtorigin">
        <time>
          <value>2017-08-27T04:17:56.800000Z</value>
          <uncertainty>0.0</uncertainty>
        </time>
        <latitude>
          <value>-1.25</value>
          <uncertainty>0.0</uncertainty>
        </latitude>
        <longitude>
          <value>148.15</value>
          <uncertainty>0.0</uncertainty>
        </longitude>
        <depth>
          <value>15900.0</value>
          <uncertainty>300.0</uncertainty>
        </depth>
        <depthType>from moment tensor inversion</depthType>
        <timeFixed>false</timeFixed>
        <epicenterFixed>false</epicenterFixed>
        <type>centroid</type>
        <creationInfo>
          <agencyID>GCMT</agencyID>
          <version>V10</version>
        </creationInfo>
      </origin>
      <magnitude publicID="smi:local/ndk/C201708270417A/magnitude#moment_mag">
        <mag>
          <value>6.31</value>
        </mag>
        <type>Mwc</type>
        <originID>smi:local/ndk/C201708270417A/origin#cmtorigin</originID>
        <creationInfo>
          <agencyID>GCMT</agencyID>
          <version>V10</version>
        </creationInfo>
      </magnitude>
      <magnitude publicID="smi:local/ndk/C201708270417A/magnitude#mb">
        <mag>
          <value>0.0</value>
        </mag>
        <type>mb</type>
        <comment id="smi:local/ndk/C201708270417A/comment#mb_magnitude">
          <text>Reported magnitude in NDK file. Most likely 'mb'.</text>
        </comment>
      </magnitude>
      <magnitude publicID="smi:local/ndk/C201708270417A/magnitude#MS">
        <mag>
          <value>6.3</value>
        </mag>
        <type>MS</type>
        <comment id="smi:local/ndk/C201708270417A/comment#MS_magnitude">
          <text>Reported magnitude in NDK file. Most likely 'MS'.</text>
        </comment>
      </magnitude>
      <focalMechanism publicID="smi:local/ndk/C201708270417A/focal_mechanism">
        <nodalPlanes>
          <nodalPlane1>
            <strike>
              <value>317.0</value>
            </strike>
            <dip>
              <value>85.0</value>
            </dip>
            <rake>
              <value>-179.0</value>
            </rake>
          </nodalPlane1>
          <nodalPlane2>
            <strike>
              <value>227.0</value>
            </strike>
            <dip>
              <value>89.0</value>
            </dip>
            <rake>
              <value>-5.0</value>
            </rake>
          </nodalPlane2>
        </nodalPlanes>
        <principalAxes>
          <tAxis>
            <azimuth>
              <value>272.0</value>
            </azimuth>
            <plunge>
              <value>3.0</value>
            </plunge>
            <length>
              <value>3.936e+18</value>
            </length>
          </tAxis>
          <pAxis>
            <azimuth>
              <value>182.0</value>
            </azimuth>
            <plunge>
              <value>4.0</value>
            </plunge>
            <length>
              <value>-3.316e+18</value>
            </length>
          </pAxis>
          <nAxis>
            <azimuth>
              <value>40.0</value>
            </azimuth>
            <plunge>
              <value>85.0</value>
            </plunge>
            <length>
              <value>-6.26e+17</value>
            </length>
          </nAxis>
        </principalAxes>
        <momentTensor publicID="smi:local/ndk/C201708270417A/momenttensor">
          <derivedOriginID>smi:local/ndk/C201708270417A/origin#cmtorigin</derivedOriginID>
          <dataUsed>
            <waveType>body waves</waveType>
            <stationCount>170</stationCount>
            <componentCount>402</componentCount>
            <shortestPeriod>40.0</shortestPeriod>
          </dataUsed>
          <dataUsed>
            <waveType>surface waves</waveType>
            <stationCount>173</stationCount>
            <componentCount>404</componentCount>
            <shortestPeriod>50.0</shortestPeriod>
          </dataUsed>
          <dataUsed>
            <waveType>mantle waves</waveType>
            <stationCount>154</stationCount>
            <componentCount>290</componentCount>
            <shortestPeriod>125.0</shortestPeriod>
          </dataUsed>
          <scalarMoment>
            <value>3.626e+18</value>
          </scalarMoment>
          <tensor>
            <Mrr>
              <value>-6.26e+17</value>
              <uncertainty>1.5e+16</uncertainty>
            </Mrr>
            <Mtt>
              <value>-3.29e+18</value>
              <uncertainty>1.6e+16</uncertainty>
            </Mtt>
            <Mpp>
              <value>3.91e+18</value>
              <uncertainty>1.7e+16</uncertainty>
            </Mpp>
            <Mrt>
              <value>2.14e+17</value>
              <uncertainty>4.3e+16</uncertainty>
            </Mrt>
            <Mrp>
              <value>2.59e+17</value>
              <uncertainty>4.4e+16</uncertainty>
            </Mrp>
            <Mtp>
              <value>2.77e+17</value>
              <uncertainty>1.4e+16</uncertainty>
            </Mtp>
          </tensor>
          <sourceTimeFunction>
            <type>triangle</type>
            <duration>7.0</duration>
          </sourceTimeFunction>
          <inversionType>zero trace</inversionType>
          <creationInfo>
            <agencyID>GCMT</agencyID>
            <version>V10</version>
          </creationInfo>
        </momentTensor>
        <comment id="smi:local/ndk/C201708270417A/comment#cmt_type">
          <text>CMT Analysis Type: Standard</text>
        </comment>
        <comment id="smi:local/ndk/C201708270417A/comment#cmt_timestamp">
          <text>CMT Timestamp: S-20171030010012</text>
        </comment>
        <creationInfo>
          <agencyID>GCMT</agencyID>
          <version>V10</version>
        </creationInfo>
      </focalMechanism>
    </event>
  </eventParameters>
</q:quakeml>
