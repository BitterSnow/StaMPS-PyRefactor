#author：sxp
import scipy.io as scio
import ogr,os
import numpy as np
import osr
import datetime
dataPath=r"D:\work\生产任务\S316\vel"
#load data file
ps2 = scio.loadmat(dataPath+r"\ps2.mat")
ps_plot_v = scio.loadmat(dataPath+r"\ps_plot_v-do.mat")
# ps_plot_v = scio.loadmat(dataPath+r"\ps_plot_v.mat")
dataTS=scio.loadmat(dataPath+r"\ps_plot_ts_v.mat")

lonlat = ps2['lonlat']
ph_disp = ps_plot_v['ph_disp']
phmmData=dataTS['ph_mm']
masterDay=dataTS['master_day']
days=dataTS['day']
daysAll= []
ltDate=[]
days=dataTS['day']
daysAll= []
baseDay=736942
name='cqbFull'
baseDate=datetime.datetime.strptime('20170904','%Y%m%d')
masterDay=dataTS['master_day']
masterDate=baseDate+datetime.timedelta(days=(int(masterDay)-baseDay))
masterDateStr=masterDate.strftime('%Y%m%d')
outprjFilePath=dataPath+"\\"+name+".prj"
for singleDate in dataTS['day']:
    daysAll.append(int(singleDate[0]))
daysAll.append(int(masterDay[0]))
daysAll.sort()
for singleDate in daysAll:
    dateChange=int(singleDate)-baseDay
    newDate=baseDate+datetime.timedelta(days=dateChange)
    ltDate.append(newDate.strftime("%Y%m%d"))
indexofMaster=ltDate.index(masterDateStr)
phmm=np.insert(phmmData,indexofMaster,0,axis=1)
imageNum=np.shape(phmm)[1]
fieldType = ogr.OFTReal
fieldNameVel = "vel"
outSHPfn = dataPath+'\\'+name+'.shp'
shpDriver = ogr.GetDriverByName("ESRI Shapefile")
for ii in range(0,np.shape(phmm)[0]):
    phmm[ii,:]=phmm[ii,:]-phmm[ii,0]
if os.path.exists(outSHPfn):
    shpDriver.DeleteDataSource(outSHPfn)
outDataSource = shpDriver.CreateDataSource(outSHPfn)
outLayer = outDataSource.CreateLayer(outSHPfn, geom_type=ogr.wkbPoint)
#Create Vel field
idFieldVel = ogr.FieldDefn(fieldNameVel, fieldType)
outLayer.CreateField(idFieldVel)

#Create TS field
for ii in range(0,imageNum):
    date = str(ltDate[ii])
    dateFieldName ='D'+ date[0:4] + date[4:6] + date[6:8]
    idTSField = ogr.FieldDefn(dateFieldName,fieldType)
    outLayer.CreateField(idTSField)

featureDefn = outLayer.GetLayerDefn()
outFeature = ogr.Feature(featureDefn)
date0='D'+ ltDate[0][0:4] +ltDate[0][4:6] +  ltDate[0][6:8]
for ii in range(0,lonlat.shape[0]):
    px = lonlat[ii,0]
    py = lonlat[ii,1]
    point = ogr.Geometry(ogr.wkbPoint)
    point.AddPoint(float(px),float(py))
    outFeature.SetGeometry(point)
    outFeature.SetField(fieldNameVel, float(ph_disp[ii]))
    outFeature.SetField(date0, 0.0)
    for jj in range(0,imageNum):
        date = str(ltDate[jj])
        dateFieldName = 'D'+ date[0:4] + date[4:6] + date[6:8]
        outFeature.SetField(dateFieldName, float(phmm[ii][jj]))
    outLayer.CreateFeature(outFeature)
spatialRef = osr.SpatialReference()
spatialRef.ImportFromEPSG(4326)
spatialRef.MorphToESRI()
file = open(outprjFilePath, 'w')
file.write(spatialRef.ExportToWkt())
file.close()
print("done!")