#include <stdio.h>
#include <assert.h>
#include <errno.h>
#include <string.h>      // strerror_r

#include <set>
#include <string>

#include "trace.h"

// define TRACING to compile these functions, e.g.,
// gcc -DTRACING -c trace.cpp
#ifdef TRACING

#include <cuda_runtime.h>
#include "magma_internal.h"
#include "magmablas_v1.h"

// set TRACE_METHOD = 2 to record start time as
// later of CPU time and previous event's end time.
// set TRACE_METHOD = 1 to record start time using CUDA event.
#define TRACE_METHOD 2


/******************************************************************************/
struct event_log
{
    int    ncore;
    int    cpu_id   [ MAX_CORES ];
    double cpu_first;
    double cpu_start[ MAX_CORES ][ MAX_EVENTS ];
    double cpu_end  [ MAX_CORES ][ MAX_EVENTS ];
    char   cpu_tag  [ MAX_CORES ][ MAX_EVENTS ][ MAX_LABEL_LEN ];
    char   cpu_label[ MAX_CORES ][ MAX_EVENTS ][ MAX_LABEL_LEN ];
    
    int           ngpu;
    int           nqueue;
    magma_queue_t queues   [ MAX_GPU_QUEUES ];
    int           gpu_id   [ MAX_GPU_QUEUES ];
    magma_event_t gpu_first[ MAX_GPU_QUEUES ];
#if TRACE_METHOD == 2
    double        gpu_start[ MAX_GPU_QUEUES ][ MAX_EVENTS ];
#else
    magma_event_t gpu_start[ MAX_GPU_QUEUES ][ MAX_EVENTS ];
#endif
    magma_event_t gpu_end  [ MAX_GPU_QUEUES ][ MAX_EVENTS ];
    char          gpu_tag  [ MAX_GPU_QUEUES ][ MAX_EVENTS ][ MAX_LABEL_LEN ];
    char          gpu_label[ MAX_GPU_QUEUES ][ MAX_EVENTS ][ MAX_LABEL_LEN ];
};

// global log object
struct event_log glog;


/******************************************************************************/
void trace_init( int ncore, int ngpu, int nqueue, magma_queue_t* queues )
{
    if ( ncore > MAX_CORES ) {
        fprintf( stderr, "Error in trace_init: ncore %d > MAX_CORES %d\n",
                 ncore, MAX_CORES );
        exit(1);
    }
    if ( ngpu*nqueue > MAX_GPU_QUEUES ) {
        fprintf( stderr, "Error in trace_init: (ngpu=%d)*(nqueue=%d) > MAX_GPU_QUEUES=%d\n",
                 ngpu, nqueue, MAX_GPU_QUEUES );
        exit(1);
    }
    
    glog.ncore  = ncore;
    glog.ngpu   = ngpu;
    glog.nqueue = nqueue;
    
    // initialize ID = 0
    for( int core = 0; core < ncore; ++core ) {
        glog.cpu_id[core] = 0;
    }
    for( int dev = 0; dev < ngpu; ++dev ) {
        for( int s = 0; s < nqueue; ++s ) {
            int t = dev*glog.nqueue + s;
            glog.gpu_id[t] = 0;
            glog.queues[t] = queues[t];
        }
        magma_setdevice( dev );
        magma_device_sync();
    }
    // now that all GPUs are sync'd, record start time
    for( int dev = 0; dev < ngpu; ++dev ) {
        magma_setdevice( dev );
        for( int s = 0; s < nqueue; ++s ) {
            int t = dev*glog.nqueue + s;
            magma_event_create( &glog.gpu_first[t] );
            magma_event_record(  glog.gpu_first[t], glog.queues[t] );
        }
    }
    // sync again
    for( int dev = 0; dev < ngpu; ++dev ) {
        magma_setdevice( dev );
        magma_device_sync();
    }
    glog.cpu_first = magma_wtime();
}


/******************************************************************************/
void trace_cpu_start( int core, const char* tag, const char* lbl )
{
    int id = glog.cpu_id[core];
    glog.cpu_start[core][id] = magma_wtime();
    magma_strlcpy( glog.cpu_tag  [core][id], tag, MAX_LABEL_LEN );
    magma_strlcpy( glog.cpu_label[core][id], lbl, MAX_LABEL_LEN );
}


/******************************************************************************/
void trace_cpu_end( int core )
{
    int id = glog.cpu_id[core];
    glog.cpu_end[core][id] = magma_wtime();
    if ( id+1 < MAX_EVENTS ) {
        glog.cpu_id[core] = id+1;
    }
    else {
        fprintf( stderr, "Error in %s: not enough CPU events.\n", __func__ );
    }
}


/******************************************************************************/
void trace_gpu_start( int dev, int s, const char* tag, const char* lbl )
{
    int t = dev*glog.nqueue + s;
    int id = glog.gpu_id[t];
#if TRACE_METHOD == 2
    glog.gpu_start[t][id] = magma_wtime();
#else
    magma_event_create( &glog.gpu_start[t][id] );
    magma_event_record(  glog.gpu_start[t][id], glog.queues[t] );
#endif
    magma_strlcpy( glog.gpu_tag  [t][id], tag, MAX_LABEL_LEN );
    magma_strlcpy( glog.gpu_label[t][id], lbl, MAX_LABEL_LEN );
}


/******************************************************************************/
void trace_gpu_end( int dev, int s )
{
    int t = dev*glog.nqueue + s;
    int id = glog.gpu_id[t];
    magma_event_create( &glog.gpu_end[t][id] );
    magma_event_record(  glog.gpu_end[t][id], glog.queues[t] );
    if ( id+1 < MAX_EVENTS ) {
        glog.gpu_id[t] = id+1;
    }
    else {
        fprintf( stderr, "Error in %s: not enough GPU events (dev %d, queue %d).\n", __func__, dev, s );
    }
}


/******************************************************************************/
void trace_finalize( const char* filename, const char* cssfile )
{
    // these are all in SVG "pixels"
    double xscale = 200.; // pixels per second
    double height = 20.;  // of each row
    double margin =  5.;  // page margin and between some elements
    double space  =  2.;  // between rows
    double pad    =  5.;  // around text
    double label  = 75.;  // width of "CPU:", "GPU:" labels
    double left   = 2*margin + label;
    double xtick  = 0.5;  // interval of xticks (in seconds)
    char buf[ 1024 ];
    
    // sync devices
    for( int dev = 0; dev < glog.ngpu; ++dev ) {
        magma_setdevice( dev );
        magma_device_sync();
    }
    double time = magma_wtime() - glog.cpu_first;
    
    FILE* trace_file = fopen( filename, "w" );
    if ( trace_file == NULL ) {
        strerror_r( errno, buf, sizeof(buf) );
        fprintf( stderr, "Can't open file '%s': %s (%d)\n", filename, buf, errno );
        return;
    }
    fprintf( stderr, "writing trace to '%s'\n", filename );
    
    // row for each CPU and GPU/queue (with space between), time scale, legend
    // 4 margins: at top, above time scale, above legend, at bottom
    int h = (int)( (glog.ncore + glog.ngpu*glog.nqueue)*(height + space) - space + 2*height + 4*margin );
    int w = (int)( left + time*xscale + margin );
    fprintf( trace_file,
             "<?xml version=\"1.0\" standalone=\"no\"?>\n"
             "<!DOCTYPE svg PUBLIC \"-//W3C//DTD SVG 1.1//EN\"\n"
             "    \"http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd\">\n\n"
             "<svg version=\"1.1\" baseProfile=\"full\"\n"
             "    xmlns=\"http://www.w3.org/2000/svg\"\n"
             "    xmlns:inkscape=\"http://www.inkscape.org/namespaces/inkscape\"\n"
             "    viewBox=\"0 0 %d %d\" width=\"%d\" height=\"%d\" preserveAspectRatio=\"none\">\n\n",
             w, h, w, h );
    
    // Inkscape does not currently (Jan 2012) support external CSS;
    // see http://wiki.inkscape.org/wiki/index.php/CSS_Support
    // So embed CSS file here
    FILE* css_file = fopen( cssfile, "r" );
    if ( css_file == NULL ) {
        strerror_r( errno, buf, sizeof(buf) );
        fprintf( stderr, "Can't open file '%s': %s; skipping CSS\n", cssfile, buf );
    }
    else {
        fprintf( trace_file, "<style type=\"text/css\">\n" );
        while( fgets( buf, sizeof(buf), css_file ) != NULL ) {
            fputs( buf, trace_file );
        }
        fclose( css_file );
        fprintf( trace_file, "</style>\n\n" );
    }
    
    // format takes: x, y, width, height, class (tag), id (label)
    const char* format =
        "<rect x=\"%8.3f\" y=\"%4.0f\" width=\"%8.3f\" height=\"%2.0f\" class=\"%-8s\" inkscape:label=\"%s\"/>\n";
    
    // accumulate unique legend entries
    std::set< std::string > legend;
    
    // output CPU events
    double top = margin;
    for( int core = 0; core < glog.ncore; ++core ) {
        if ( glog.cpu_id[core] > MAX_EVENTS ) {
            glog.cpu_id[core] += 1;  // count last event
            fprintf( stderr, "WARNING: trace on core %d, reached limit of %d events; output will be truncated.\n",
                     core, glog.cpu_id[core] );
        }
        fprintf( trace_file, "<!-- core %d, nevents %d -->\n", core, glog.cpu_id[core] );
        fprintf( trace_file, "<g inkscape:groupmode=\"layer\" inkscape:label=\"core %d\">\n", core );
        fprintf( trace_file, "<text x=\"%8.3f\" y=\"%4.0f\" width=\"%4.0f\" height=\"%2.0f\">CPU %d:</text>\n",
                 margin,
                 top + height - pad,
                 label, height,
                 core );
        for( int i = 0; i < glog.cpu_id[core]; ++i ) {
            double start  = glog.cpu_start[core][i] - glog.cpu_first;
            double end    = glog.cpu_end  [core][i] - glog.cpu_first;
            fprintf( trace_file, format,
                     left + start*xscale,
                     top,
                     (end - start)*xscale,
                     height,
                     glog.cpu_tag[core][i],
                     glog.cpu_label[core][i] );
            legend.insert( glog.cpu_tag[core][i] );
        }
        top += (height + space);
        fprintf( trace_file, "</g>\n\n" );
    }
    
    // output GPU events
    for( int dev = 0; dev < glog.ngpu; ++dev ) {
        for( int s = 0; s < glog.nqueue; ++s ) {
            int t = dev*glog.nqueue + s;
            if ( glog.gpu_id[t] >= MAX_EVENTS-1 ) {
                glog.gpu_id[t] += 1;  // count last event
                fprintf( stderr, "WARNING: trace on gpu %d/queue %d reached limit of %d events; output will be truncated.\n",
                         dev, s, glog.gpu_id[t] );
            }
            fprintf( trace_file, "<!-- gpu %d, queue %d (t %d), nevents %d -->\n", dev, s, t, glog.gpu_id[t] );
            fprintf( trace_file, "<g inkscape:groupmode=\"layer\" inkscape:label=\"gpu %d queue %d\">\n", dev, s );
            fprintf( trace_file, "<text x=\"%8.3f\" y=\"%4.0f\" width=\"%4.0f\" height=\"%2.0f\">GPU %d (s%d):</text>\n",
                     margin,
                     margin + (dev*glog.nqueue + s + glog.ncore)*(height + space) + height - pad,
                     label, height,
                     dev, s );
            magma_setdevice( dev );
            float start, end;
            for( int i = 0; i < glog.gpu_id[t]; ++i ) {
                #if TRACE_METHOD == 2
                    start = glog.gpu_start[t][i] - glog.cpu_first;
                    if ( i > 0 ) {
                        // later of task's CPU start time and previous task's end time
                        start = max( start, end );
                    }
                #else
                    cudaEventElapsedTime( &start, glog.gpu_first[t], glog.gpu_start[t][i] );
                    start *= 1e-3;  // ms to seconds
                #endif
                cudaEventElapsedTime( &end, glog.gpu_first[t], glog.gpu_end  [t][i] );
                end   *= 1e-3;  // ms to seconds
                fprintf( trace_file, format,
                         left + start*xscale,
                         top,
                         (end - start)*xscale,
                         height,
                         glog.gpu_tag[t][i],
                         glog.gpu_label[t][i] );
                legend.insert( glog.gpu_tag[t][i] );
            }
            top += (height + space);
            fprintf( trace_file, "</g>\n\n" );
        }
    }
    
    // output time scale
    top += (-space + margin);
    fprintf( trace_file, "<g inkscape:groupmode=\"layer\" inkscape:label=\"scale\">\n" );
    fprintf( trace_file, "<text x=\"%8.1f\" y=\"%4.0f\" width=\"%2.0f\" height=\"%2.0f\">Time (sec):</text>\n",
             margin, top + height - pad, label, height );
    for( double s=0; s < time; s += xtick ) {
        fprintf( trace_file,
            "<line x1=\"%8.1f\" y1=\"0\" x2=\"%8.1f\" y2=\"%4.0f\"/>"
            "<text x=\"%8.1f\" y=\"%4.0f\">%4.1f</text>\n",
            left + s*xscale,
            left + s*xscale,
            top,
            left + s*xscale,
            top + height - pad,
            s );
    }
    fprintf( trace_file, "</g>\n\n" );
    top += (height + margin);
    
    // output legend
    fprintf( trace_file, "<g inkscape:groupmode=\"layer\" inkscape:label=\"legend\">\n" );
    fprintf( trace_file, "<text x=\"%8.1f\" y=\"%4.0f\" width=\"%2.0f\" height=\"%2.0f\">Legend:</text>\n",
             margin, top + height - pad, label, height );
    double x=left;
    for( std::set<std::string>::iterator it=legend.begin(); it != legend.end(); ++it ) {
        fprintf( trace_file,
            "<rect x=\"%8.1f\" y=\"%4.0f\" width=\"%2.0f\" height=\"%2.0f\" class=\"%s\"/>"
            "<text x=\"%8.1f\" y=\"%4.0f\" width=\"%2.0f\" height=\"%2.0f\">%s</text>",
            x,       top,                label, height, (*it).c_str(),
            x + pad, top + height - pad, label, height, (*it).c_str() );
        x += label + margin;
    }
    fprintf( trace_file, "</g>\n\n" );
    
    fprintf( trace_file, "</svg>\n" );
    
    fclose( trace_file );
}

#endif // TRACING
