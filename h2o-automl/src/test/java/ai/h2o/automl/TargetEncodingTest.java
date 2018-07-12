package ai.h2o.automl;

import hex.FrameSplitter;
import hex.Model;
import hex.genmodel.utils.DistributionFamily;
import hex.tree.gbm.GBM;
import hex.tree.gbm.GBMModel;
import org.junit.*;
import water.H2O;
import water.fvec.Frame;
import water.fvec.TestFrameBuilder;
import water.fvec.Vec;
import water.rapids.ast.prims.mungers.AstGroup;
import water.util.FrameUtils;
import water.util.IcedHashMap;
import water.util.TwoDimTable;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;
import static water.util.FrameUtils.generateNumKeys;

public class TargetEncodingTest extends TestUtil{


  @BeforeClass public static void setup() {
    stall_till_cloudsize(1);
  }

  private Frame fr = null;

  @Before
  public void beforeEach() {
        System.out.println("Before each setup");
    }

  @Ignore
  @Test public void TitanicDemoWithTargetEncodingTest() {
    Frame fr=null;
    try {
      fr = parse_test_file("./smalldata/gbm_test/titanic.csv");


      fr = FrameUtils.categoricalEncoder(fr, new String[]{},
              Model.Parameters.CategoricalEncodingScheme.LabelEncoder, null, -1);

      // Splitting phase

      double[] ratios  = ard(0.7f, 0.1f, 0.1f);
      Frame[] splits  = null;
      long numberOfRows = fr.numRows();
      FrameSplitter fs = new FrameSplitter(fr, ratios, generateNumKeys(fr._key, ratios.length+1), null);
      H2O.submitTask(fs).join();
      splits = fs.getResult();
      Frame train = splits[0];
      Frame valid = splits[1];
      Frame te_holdout = splits[2];
      Frame test = splits[3];
      long l = train.numRows();
      double v = Math.floor(numberOfRows * 0.7);
      assert l == v;
      assert splits.length == 4;

      String[] colNames = train.names();

      //myX <- setdiff(colnames(train), c("survived", "name", "ticket", "boat", "body"))


      // Building default GBM
      GBMModel gbm = null;
      Frame fr2 = null;

      GBMModel.GBMParameters parms = new GBMModel.GBMParameters();
      parms._train = train._key;
      parms._response_column = "survived";
      parms._ntrees = 2;
      parms._max_depth = 3;
      parms._distribution = DistributionFamily.AUTO;
      parms._keep_cross_validation_predictions=true;
      GBM job = new GBM(parms);
      gbm = job.trainModel().get();

      Assert.assertTrue(job.isStopped());

      fr2 = gbm.score(fr);

    } finally {
      if(fr != null) fr.remove();
    }
  }

    @Test(expected = IllegalStateException.class)
    public void targetEncoderPrepareEncodingFrameValidationDataIsNotNullTest() {

        TargetEncoder tec = new TargetEncoder();
        String[] teColumns = {"0"};

        //TODO rewrite with try/catch
        tec.prepareEncodingMap(null, teColumns, "2", null);

    }

    @Test(expected = IllegalStateException.class)
    public void targetEncoderPrepareEncodingFrameValidationTEColumnsIsNotEmptyTest() {

        TargetEncoder tec = new TargetEncoder();
        String[] teColumns = {};

        tec.prepareEncodingMap(null, teColumns, "2", null);

    }

    @Test(expected = IllegalStateException.class)
    public void targetEncoderPrepareEncodingFrameValidationTest() {

        //TODO test other validation checks

    }

  @Ignore
  @Test public void groupByWithAstGroupTest() {
      Frame fr=null;
      try {
          fr = parse_test_file("./smalldata/gbm_test/titanic.csv");
          int[] gbCols = {1};
          IcedHashMap<AstGroup.G, String> gss = AstGroup.doGroups(fr, gbCols, AstGroup.aggNRows());
          IcedHashMap<AstGroup.G, String> g = gss;
      }
      finally {
          if(fr != null) fr.remove();
      }
  }

    @Test
    public void targetEncoderFilterOutNAsTest() {

        fr = new TestFrameBuilder()
                .withName("testFrame")
                .withColNames("ColA", "ColB", "ColC")
                .withVecTypes(Vec.T_NUM, Vec.T_NUM, Vec.T_STR)
                .withDataForCol(0, ard(1, 1))
                .withDataForCol(1, ard(1, 1))
                .withDataForCol(2, ar(null, "6"))
                .build();
        TargetEncoder tec = new TargetEncoder();
        Frame result = tec.filterOutNAsFromTargetColumn(fr, "2");
        assertEquals(1L, result.numRows());

    }

    @Test
    public void allTEColumnsAreCategoricalTest() {

        TestFrameBuilder baseBuilder = new TestFrameBuilder()
                .withName("testFrame")
                .withColNames("ColA", "ColB", "ColC")
                .withDataForCol(0, ar("1", "0"))
                .withDataForCol(2, ar("1", "6"));

        TargetEncoder tec = new TargetEncoder();
        String[] teColumns = {"0", "1"};

        fr = baseBuilder
                .withDataForCol(1, ar(0, 1))
                .withVecTypes(Vec.T_CAT, Vec.T_NUM, Vec.T_CAT)
                .build();
        try {
            tec.prepareEncodingMap(fr, teColumns, "2", null);
        } catch (IllegalStateException ex) {
            assertEquals(ex.getMessage(), "Argument 'columnsToEncode' should contain only names of categorical columns");
        }

        fr = baseBuilder
                .withDataForCol(1, ar("a", "b"))
                .withVecTypes(Vec.T_CAT, Vec.T_CAT, Vec.T_CAT)
                .build();

        try {
            Frame result = tec.prepareEncodingMap(fr, teColumns, "2", null);
            String[]res = result.toTwoDimTable().getColHeaders();
            Frame r = null;
        } catch (IllegalStateException ex) {
            fail(String.format("All columns were categorical but something else went wrong: %s", ex.getMessage()));
        }
    }

    @Test
    public void targetEncoderPrepareEncodingMapTest() {
        fr = new TestFrameBuilder()
                .withName("testFrame")
                .withColNames("ColA", "ColB", "ColC", "fold_column")
                .withVecTypes(Vec.T_CAT, Vec.T_NUM, Vec.T_CAT,  Vec.T_NUM)
                .withDataForCol(0, ar("a", "b"))
                .withDataForCol(1, ard(1, 1))
                .withDataForCol(2, ar("2", "6"))
                .withDataForCol(3, ar(1, 2))
                .build();

        TargetEncoder tec = new TargetEncoder();
        String[] teColumns = {"0"};

        Frame targetEncodingMap = tec.prepareEncodingMap(fr, teColumns, "2", "3");

        TwoDimTable twoDimTable = targetEncodingMap.toTwoDimTable();
        System.out.println(twoDimTable.toString());


    }

    @Test
    public void targetEncoderKFoldHoldoutApplyingTest() {
        fr = new TestFrameBuilder()
                .withName("testFrame")
                .withColNames("ColA", "ColB", "ColC", "fold_column")
                .withVecTypes(Vec.T_CAT, Vec.T_NUM, Vec.T_CAT,  Vec.T_NUM)
                .withDataForCol(0, ar("a", "b"))
                .withDataForCol(1, ard(1, 1))
                .withDataForCol(2, ar("2", "6"))
                .withDataForCol(3, ar(1, 2))
                .build();

        TargetEncoder tec = new TargetEncoder();
        String[] teColumns = {"0"};

        Frame targetEncodingMap = tec.prepareEncodingMap(fr, teColumns, "2", "3");

        // fold_column index is changed due to grouping
        tec.applyTargetEncoding(fr, teColumns, "2", targetEncodingMap, TargetEncoder.HoldoutType.KFold, "1", false, 0);

    }

    @Test
    public void renameColumnTest() {
        fr = new TestFrameBuilder()
                .withName("testFrame")
                .withColNames("ColA", "ColB", "ColC", "fold_column")
                .withVecTypes(Vec.T_CAT, Vec.T_NUM, Vec.T_CAT,  Vec.T_NUM)
                .withDataForCol(0, ar("a", "b"))
                .withDataForCol(1, ard(1, 1))
                .withDataForCol(2, ar("2", "6"))
                .withDataForCol(3, ar(1, 2))
                .build();

        TargetEncoder tec = new TargetEncoder();
        String indexOfColumnToRename = "0";
        String newName = "NewColA";
        Frame renamedFrame = tec.renameColumn(fr, indexOfColumnToRename, newName);


        TwoDimTable twoDimTable = renamedFrame.toTwoDimTable();
        System.out.println(twoDimTable.getColHeaders()[Integer.parseInt(indexOfColumnToRename)]);
        assertEquals( twoDimTable.getColHeaders()[Integer.parseInt(indexOfColumnToRename)], newName);
    }

    @Test
    public void transformBinaryTargetColumnTest() {
        fr = new TestFrameBuilder()
                .withName("testFrame")
                .withColNames("ColA", "ColB", "ColC", "fold_column")
                .withVecTypes(Vec.T_CAT, Vec.T_NUM, Vec.T_CAT,  Vec.T_NUM)
                .withDataForCol(0, ar("a", "b"))
                .withDataForCol(1, ard(1, 1))
                .withDataForCol(2, ar("2", "6"))
                .withDataForCol(3, ar(1, 2))
                .build();

        TargetEncoder tec = new TargetEncoder();

        TwoDimTable twoDimTable = fr.toTwoDimTable();
        System.out.println(twoDimTable.toString());

        Frame res = tec.transformBinaryTargetColumn(fr, 2);

        TwoDimTable twoDimTable2 = res.toTwoDimTable();
        System.out.println(twoDimTable2.toString());

        // TODO Different levels are transformed into the same numeric value.
        assertEquals(false, true);

    }

    @Test
    public void targetEncoderGetOutOfFoldDataTest() {
        fr = new TestFrameBuilder()
                .withName("testFrame")
                .withColNames("ColA", "ColB")
                .withVecTypes(Vec.T_NUM, Vec.T_NUM)
                .withDataForCol(0, ard(5, 6, 7))
                .withDataForCol(1, ard(1, 2, 3))
                .build();

        TargetEncoder tec = new TargetEncoder();

        Frame outOfFoldData = tec.getOutOfFoldData(fr, "1", 1);
        TwoDimTable twoDimTable = outOfFoldData.toTwoDimTable();

        assertEquals(6L, twoDimTable.get(5, 0));
        assertEquals(7L, twoDimTable.get(6, 0));

        Frame outOfFoldData2 = tec.getOutOfFoldData(fr, "1", 2);
        TwoDimTable twoDimTable2 = outOfFoldData2.toTwoDimTable();

        assertEquals(5L, twoDimTable2.get(5, 0));
        assertEquals(7L, twoDimTable2.get(6, 0));

    }

    @After
    public void afterEach() {
        System.out.println("After each setup");
        // TODO in checkLeakedKeys method from TestUntil we are purging store anyway. So maybe we should add default cleanup? or we need to inform developer about specific leakages?
        H2O.STORE.clear();
    }
}
